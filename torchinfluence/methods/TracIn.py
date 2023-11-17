from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader, Dataset, Subset


class GradientSimilarity:
    """
    GradientSimilarity is a class that computes the similarity scores between the test dataset and the train dataset using
    the dot product of their gradients. It also computes the gradients of the loss function with respect to the
    model parameters for a given dataset.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
        device: str = "cpu",
        parameter_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes a new instance of the GradientSimilarity class.

        Args:
            model (torch.nn.Module): The PyTorch model to compute the gradients for.
            loss_fn (Callable): The loss function to compute the gradients for.
            device (str): The device to use for computation.
            selected_params (Optional[List[str]]): The selected parameters to compute the gradients for.
        """
        self.device = device
        self.model = model.to(device)

        self.params = dict(model.named_parameters())
        if parameter_names is not None:
            self.params = {name: param for name, param in self.params.items() if name in parameter_names}

        self.loss_fn = loss_fn

    def _compute_loss(self, params, inputs, targets):
        """
        Computes the loss function for a given set of parameters, inputs and targets.

        Args:
            params (dict): The parameters to compute the loss function for.
            inputs (torch.Tensor): The inputs to compute the loss function for.
            targets (torch.Tensor): The targets to compute the loss function for.

        Returns:
            torch.Tensor: The loss function value.
        """
        prediction = functional_call(self.model, params, (inputs,))
        return self.loss_fn(prediction, targets)

    def dataset_gradients(self, dataset: Dataset) -> torch.Tensor:
        """
        Computes the gradients of the loss function with respect to the model parameters
        for a given dataset.

        Args:
            dataset (Dataset): the dataset to compute the gradients for.

        Returns:
            torch.Tensor: a tensor containing the gradients of the loss function with respect
            to the model parameters, stacked horizontally.
        """
        compute_grads = vmap(grad(self._compute_loss), in_dims=(None, 0, 0), chunk_size=512)
        grads = compute_grads(self.params, dataset[:][0].to(self.device), dataset[:][1].to(self.device))
        grads = torch.hstack([g.reshape(len(dataset), -1) for g in list(grads.values())])  # (len(dataset), num_params)
        return grads

    def score(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        subset_ids: Union[str, List[int] | None] = {"train": None, "test": None},
        normalize: Optional[bool] = False,
        chunk_size: Optional[int] = 512,
    ) -> np.ndarray:
        """
        Computes the similarity scores between the test dataset and the train dataset using
        the dot product of their gradients.

        Args:
            train_dataset (Dataset): The dataset used for training.
            test_dataset (Dataset): The dataset used for testing.
            subset_ids (Union[str, List[int] | None], optional): The ids of the subset of the train and test datasets
                                                                 to use for computing the scores. Defaults to
                                                                 {"train": None, "test": None}.
            normalize (bool, optional): Whether to normalize the scores. Defaults to False.
            chunk_size (int, optional): The size of the chunks of the train dataset to use for computing the
                                        scores. Defaults to 512.

        Returns:
            np.ndarray: A 2D array of shape (len(test_dataset), len(train_dataset)) containing the similarity scores
                        between the test dataset and the train dataset.
        """
        datasets = {"train": train_dataset, "test": test_dataset}

        subset_ids = {split: subset_ids[split] if split in subset_ids.keys() else None for split in ["train", "test"]}

        for split in subset_ids.keys():
            if subset_ids[split] is not None:
                datasets[split] = Subset(datasets[split], subset_ids[split])

        train_dataset, test_dataset = datasets["train"], datasets["test"]

        scores = torch.zeros((len(test_dataset), len(train_dataset))).to(self.device)
        test_grads = self.dataset_gradients(test_dataset)

        for chunk_ids in DataLoader(range(0, len(train_dataset)), batch_size=chunk_size):
            train_dataset_chunk = Subset(train_dataset, chunk_ids)

            if subset_ids["train"] is not None:
                train_dataset_chunk = train_dataset_chunk.dataset

            train_grads = self.dataset_gradients(train_dataset_chunk)
            scores[:, chunk_ids] = torch.matmul(test_grads, train_grads.T)

            if normalize:
                test_norm, train_norm = torch.norm(test_grads, dim=1)[:, None], torch.norm(train_grads, dim=1)[None, :]
                norm_prod = (test_norm * train_norm).clamp(min=1e-8)
                scores[:, chunk_ids] = scores[:, chunk_ids] / norm_prod

        return scores.cpu().detach().numpy()

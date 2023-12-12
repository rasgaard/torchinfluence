from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm


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
        parameter_subset: Optional[List[str]] = None,
    ):
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
        if parameter_subset is not None:
            self.params = {name: param for name, param in self.params.items() if name in parameter_subset}

        self.loss_fn = loss_fn
        self.chunk_size = 512

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

    def dataset_gradients(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the gradients of the loss function with respect to the model parameters
        for a given dataset.

        Args:
            inputs (torch.Tensor): The inputs to compute the gradients for.
            targets (torch.Tensor): The targets to compute the gradients for.
        Returns:
            torch.Tensor: a tensor containing the gradients of the loss function with respect
            to the model parameters, stacked horizontally.
        """

        if (len(inputs.shape) == 3) and (inputs.shape[1] == 1):
            inputs = inputs.squeeze(1)

        compute_grads = vmap(grad(self._compute_loss), in_dims=(None, 0, 0), chunk_size=self.chunk_size)

        inputs, targets = inputs.to(self.device), targets.to(self.device)

        grads = compute_grads(self.params, inputs, targets)
        grads = torch.hstack([g.reshape(len(inputs), -1) for g in list(grads.values())])
        return grads

    def create_subset(self, train_dataset: Dataset, test_dataset: Dataset, subset_ids: Dict[str, List[int]]):
        """
        Creates a subset of the train and test datasets.

        Args:
            train_dataset (Dataset): The original train dataset.
            test_dataset (Dataset): The original test dataset.
            subset_ids (Dict[str, List[int]]): A dictionary containing the subset ids for train and test datasets.

        Returns:
            Tuple[Dataset, Dataset]: The train and test datasets with the specified subsets.
        """
        datasets = {"train": train_dataset, "test": test_dataset}

        subset_ids = {split: subset_ids[split] if split in subset_ids.keys() else None for split in ["train", "test"]}

        for split in subset_ids.keys():
            if subset_ids[split] is not None:
                datasets[split] = Subset(datasets[split], subset_ids[split])

        train_dataset, test_dataset = datasets["train"], datasets["test"]
        return train_dataset, test_dataset

    def create_chunk_loop(self, dataset: Dataset):
        id_loader = DataLoader(range(len(dataset)), batch_size=self.chunk_size)  # todo: change to simple range
        data_loader = DataLoader(dataset, batch_size=self.chunk_size)

        chunk_loop = zip(id_loader, data_loader)

        return chunk_loop

    def score(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        subset_ids: Dict[str, List[int] | None] = {"train": None, "test": None},
        normalize: Optional[bool] = False,
        chunk_size: Optional[int] = 512,
        progress_bar: Optional[bool] = True,
    ) -> torch.Tensor:
        """
        Computes the similarity scores between the test dataset and the train dataset using
        the dot product of their gradients.

        Args:
            train_dataset (Dataset): The dataset used for training.
            test_dataset (Dataset): The dataset used for testing.
            subset_ids (Dict[str, List[int] | None], optional): The ids of the subset of the train and test datasets
                                                                 to use for computing the scores. Defaults to
                                                                 {"train": None, "test": None}.
            normalize (bool, optional): Whether to normalize the scores. Defaults to False.
            chunk_size (int, optional): The size of the chunks of the train dataset to use for computing the
                                        scores. Defaults to 512.

        Returns:
            torch.Tensor: A 2D array of shape (len(test_dataset), len(train_dataset)) containing the similarity scores
                        between the test dataset and the train dataset.
        """
        self.chunk_size = chunk_size

        train_dataset, test_dataset = self.create_subset(train_dataset, test_dataset, subset_ids)

        bar = tqdm(total=int(len(test_dataset) * len(train_dataset)), disable=not progress_bar)
        scores = torch.zeros((len(test_dataset), len(train_dataset)), dtype=torch.float16)

        test_chunk_loop = self.create_chunk_loop(test_dataset)

        # Compute gradients for the train dataset in chunks
        # and compute the similarity scores
        for test_chunk_ids, test_dataset_chunk in test_chunk_loop:
            test_inputs, test_targets = test_dataset_chunk
            test_grads = self.dataset_gradients(test_inputs, test_targets)
            test_idx_start, test_idx_end = test_chunk_ids[0], test_chunk_ids[-1] + 1

            if normalize:
                test_norm = torch.linalg.vector_norm(test_grads, axis=1, keepdims=True)

            train_chunk_loop = self.create_chunk_loop(train_dataset)
            for train_chunk_ids, train_dataset_chunk in train_chunk_loop:
                train_inputs, train_targets = train_dataset_chunk
                train_grads = self.dataset_gradients(train_inputs, train_targets)

                train_idx_start, train_idx_end = train_chunk_ids[0], train_chunk_ids[-1] + 1

                scores[test_idx_start:test_idx_end, train_idx_start:train_idx_end] = (
                    torch.matmul(test_grads, train_grads.T).cpu().detach().to(dtype=torch.float16)
                )

                # Normalizes dot product turning it into cosine similarity
                if normalize:
                    train_norm = torch.linalg.vector_norm(train_grads, dim=1, keepdims=True)
                    norm = (test_norm * train_norm.T).to(dtype=torch.float16).clamp(min=1e-7)
                    scores[test_idx_start:test_idx_end, train_idx_start:train_idx_end] /= norm.cpu().detach()

                bar.update(n=len(train_chunk_ids) * len(test_chunk_ids))

        return scores

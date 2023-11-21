import pytest
import torch
from torch.nn import Linear, MSELoss
from torch.utils.data import Dataset

from torchinfluence.methods import GradientSimilarity


class MockDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TestGradientSimilarity:
    @pytest.fixture
    def setup(self):
        model = Linear(10, 1)
        loss_fn = MSELoss()
        device = "cpu"
        grad_sim = GradientSimilarity(model, loss_fn, device)
        return grad_sim

    def test_dataset_gradients(self, setup):
        # Create a mock dataset
        data = torch.randn(100, 10)
        targets = torch.randn(100, 1)
        dataset = MockDataset(data, targets)

        # Compute gradients
        gradients = setup.dataset_gradients(data, targets)

        # Check that the output is a tensor
        assert isinstance(gradients, torch.Tensor)

        # Check that the output shape is correct
        num_params = sum(p.numel() for p in setup.model.parameters())
        assert gradients.shape == (100, num_params)

    def test_score(self, setup):
        # Create a mock dataset
        n_train = 100
        data1 = torch.randn(n_train, 10)
        targets1 = torch.randn(n_train, 1)
        dataset1 = MockDataset(data1, targets1)

        n_test = 100
        data2 = torch.randn(n_test, 10)
        targets2 = torch.randn(n_test, 1)
        dataset2 = MockDataset(data2, targets2)

        # Compute score
        score = setup.score(dataset1, dataset2, chunk_size=10)

        # Check that the output shape is correct
        assert score.shape[0] == n_test
        assert score.shape[1] == n_train

    def test_normalization(self, setup):
        n_train = 100
        data1 = torch.randn(n_train, 10)
        targets1 = torch.randn(n_train, 1)

        n_test = 100
        data2 = torch.randn(n_test, 10)
        targets2 = torch.randn(n_test, 1)

        dataset1 = MockDataset(data1, targets1)
        dataset2 = MockDataset(data2, targets2)

        # Compute score
        score = setup.score(dataset1, dataset2, normalize=True, chunk_size=10)

        # Check that the scores are between 0 and 1
        assert score.min() >= -1 and score.max() <= 1

    @pytest.mark.parametrize(
        "subset_ids, expected_shape",
        [
            ({"train": None, "test": None}, (100, 100)),
            ({"train": None, "test": [0, 1]}, (2, 100)),
            ({"train": [0, 1], "test": None}, (100, 2)),
            ({"train": [0, 1], "test": [0, 1]}, (2, 2)),
        ],
    )
    def test_subset_ids(self, setup, subset_ids, expected_shape):
        # Create a mock dataset
        n_train = 100
        data1 = torch.randn(n_train, 10)
        targets1 = torch.randn(n_train, 1)
        dataset1 = MockDataset(data1, targets1)

        n_test = 100
        data2 = torch.randn(n_test, 10)
        targets2 = torch.randn(n_test, 1)
        dataset2 = MockDataset(data2, targets2)

        # Compute score
        score = setup.score(dataset1, dataset2, subset_ids=subset_ids, normalize=False, chunk_size=10)

        # Check that the output shape is correct
        assert score.shape == expected_shape

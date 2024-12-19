import torch
from typing import Union
import numpy.typing as npt
import torch.nn as nn
import numpy as np
from cebra.data import SingleSessionDataset
from cebra.data.datatypes import Offset
from cebra.models import ConvolutionalModelMixin
from torch.utils.data import Dataset
import warnings

class TensorDataset(SingleSessionDataset):
    """Discrete and/or continuously indexed dataset based on torch/numpy arrays.

    If dealing with datasets sufficiently small to fit :py:func:`numpy.array` or :py:class:`torch.Tensor`, this
    dataset is sufficient---the sampling auxiliary variable should be specified with a dataloader.
    Based on whether `continuous` and/or `discrete` auxiliary variables are provided, this class
    can be used with the discrete, continuous and/or mixed data loader classes.

    Args:
        neural:
            Array of dtype ``float`` or float Tensor of shape ``(N, D)``, containing neural activity over time.
        continuous:
            Array of dtype ```float`` or float Tensor of shape ``(N, d)``, containing the continuous behavior
            variables over the same time dimension.
        discrete:
            Array of dtype ```int64`` or integer Tensor of shape ``(N, d)``, containing the discrete behavior
            variables over the same time dimension.

    Example:

        >>> import cebra.data
        >>> import torch
        >>> data = torch.randn((100, 30))
        >>> index1 = torch.randn((100, 2))
        >>> index2 = torch.randint(0,5,(100, ))
        >>> dataset = cebra.data.datasets.TensorDataset(data, continuous=index1, discrete=index2)

    """

    def __init__(self,
                 neural: Union[torch.Tensor, npt.NDArray],
                 continuous: Union[torch.Tensor, npt.NDArray] = None,
                 discrete: Union[torch.Tensor, npt.NDArray] = None,
                 offset: int = 1,
                 device: str = "cpu"):
        super().__init__(device=device)
        self.neural = self._to_tensor(neural, 'float').float()
        self.continuous = self._to_tensor(continuous, 'float')
        self.discrete = self._to_tensor(discrete, 'long')
        if self.continuous is None and self.discrete is None:
            warnings.warn(
                "You should pass at least one of the arguments 'continuous' or 'discrete'."
            )
        self.offset = offset

    def _to_tensor(self, array, dtype=None):
        if array is None:
            return None
        if isinstance(array, np.ndarray):
            if dtype == 'float':
                array = torch.from_numpy(array).float()
            elif dtype == 'long':
                array = torch.from_numpy(array).long()
            else:
                array = torch.from_numpy(array)
        return array

    @property
    def input_dimension(self) -> int:
        return self.neural.shape[1]

    @property
    def continuous_index(self):
        if self.continuous is None:
            raise NotImplementedError()
        return self.continuous

    @property
    def discrete_index(self):
        if self.discrete is None:
            raise NotImplementedError()
        return self.discrete

    def __len__(self):
        return len(self.neural)

    def __getitem__(self, index):
        index = self.expand_index(index)
        return self.neural[index].transpose(2, 1)

class SimpleTensorDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor = None, offset: Offset = Offset(1), device: torch.device = torch.device("cpu")):
        self.offset = offset
        self.device = device
        self.data = data.to(self.device)
        self.labels = labels.to(self.device) if labels is not None else None

    def __len__(self):
        return len(self.data)

    def set_offset(self, offset: Offset):
        self.offset = offset

    def __getitem__(self, index: int) -> torch.Tensor:
        left, right = self.offset.left, self.offset.right
        start = max(0, index - left)
        end = min(len(self.data), index + right)
        sequence = self.data[start:end]

        if len(sequence) < right + left:
            back = torch.zeros(right + left, sequence.shape[1], dtype=float, device=self.device)
            if index < left:
                back[left - index:] = sequence
            elif index > len(self.data) - right:
                back[:right + len(self.data) - index] = sequence
            else:
                raise ValueError("Index out of bounds. This shouldn't happen.")
            sequence = back[:]
        # Labels to go along with data if any
        with self.device:
            if self.labels is None:
                return sequence.T
            else:
                return sequence.T, self.labels[index].unsqueeze(0)
    
class SupervisedNNSolver:
    """Supervised neural network training"""

    def __init__(self,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module):
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.history = []

    def fit(self,
            loader: torch.utils.data.DataLoader,
            num_steps: int):
        """Train model for the specified number of steps.

        Args:
            loader: Data loader, which is an iterator over `cebra.data.Batch` instances.
                Each batch contains reference, positive and negative input samples.
        """

        self.model.train()
        step_idx = 0
        while True:
            for _, batch in enumerate(loader):
                self.step(batch)
                step_idx += 1
                if step_idx >= num_steps:
                    return

    def step(self, batch) -> dict:
        """Perform a single gradient update.

        Args:
            batch: The input samples in the form X, y.

        Returns:
            Dictionary containing loss.
        """
        self.optimizer.zero_grad()
        X, y = batch
        prediction = self._inference(X.float())
        loss = self.criterion(prediction, y)
        loss.backward()
        self.optimizer.step()
        self.history.append(loss.item())
        return dict(total=loss.item())

    def _inference(self, X):
        """Compute predictions for the batch."""
        prediction = self.model(X.float())
        return prediction
    
    def validation(self, valid_loader):
        self.model.eval()
        with torch.no_grad():
            for _, batch in enumerate(valid_loader):
                X, y = batch
                prediction = self._inference(X)
                loss = self.criterion(prediction, y)
                self.history.append(loss.item())
        return dict(total=loss.item())

class RDDDataset(Dataset):
    def __init__(self, df, x_label, y_label, device):
        self.df = df
        self.x_label = x_label
        self.y_label = y_label
        self.device = device
    
    def __len__(self):
        return self.df.count()
    
    def __getitem__(self, index):
        row = self.df.where(self.df.index == index).first()
        with self.device:
            X = torch.Tensor([row[self.x_label]]).float()
            y = torch.Tensor([row[self.y_label]]).float()
        return X, y
    
    def __getitems__(self, indices):
        if not isinstance(indices, list) or not all(isinstance(x, int) for x in indices):
            raise ValueError("Indices must be a list of integers")
        rows = self.df.where(self.df.index.isin(indices)).take(len(indices))
        with self.device:
            X = torch.Tensor([row[self.x_label] for row in rows]).float()
            y = torch.Tensor([row[self.y_label] for row in rows]).float()
        return X, y

class RDDDataset(Dataset):
    def __init__(self, rdd, x_label, y_label, device):
        self.rdd = rdd
        self.x_label = x_label
        self.y_label = y_label
        self.device = device
    
    def __len__(self):
        return len(self.rdd)
    
    def __getitem__(self, index):
        row = self.rdd[index]
        with self.device:
            X = torch.Tensor([row[self.x_label]]).float()
            y = torch.Tensor([row[self.y_label]]).float()
        return X, y

def transform(cls,X,session_id: int = None):
    """Transform an input sequence and return the embedding.

    Args:
        X: A numpy array or torch tensor of size ``time x dimension``.
        session_id: The session ID, an :py:class:`int` between 0 and :py:attr:`num_sessions` for
            multisession, set to ``None`` for single session.

    Returns:
        A :py:func:`numpy.array` of size ``time x output_dimension``.

    Example:

        >>> import cebra
        >>> import numpy as np
        >>> dataset =  np.random.uniform(0, 1, (1000, 30))
        >>> cebra_model = cebra.CEBRA(max_iterations=10)
        >>> cebra_model.fit(dataset)
        CEBRA(max_iterations=10)
        >>> embedding = cebra_model.transform(dataset)

    """
    model, offset = cls._select_model(X, session_id)

    # Input validation
    input_dtype = X.dtype

    with torch.no_grad():
        model.eval()

        if cls.pad_before_transform:
            X = np.pad(X, ((offset.left, offset.right - 1), (0, 0)),
                        mode="edge")
        X = torch.from_numpy(X).float().to(cls.device_)

        if isinstance(model, ConvolutionalModelMixin):
            # Fully convolutional evaluation, switch (T, C) -> (1, C, T)
            X = X.transpose(1, 0).unsqueeze(0)
            output = model(X).cpu().numpy().squeeze(0).transpose(1, 0)
        else:
            # Standard evaluation, (T, C, dt)
            output = model(X).cpu().numpy()

    if input_dtype == "float64":
        return output.astype(input_dtype)

    return output
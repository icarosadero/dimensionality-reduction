import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

def spark_rdd_to_tensor(df, key):
    """
    Convert a Spark RDD to a PyTorch tensor.

    Parameters
    ----------
    df: pyspark.RDD
        The Spark RDD to convert.
    key: str
        The key to extract from each row of the RDD.

    Returns
    -------
    tensor: torch.Tensor
        The converted tensor.
    """
    
    return torch.Tensor(list(map(lambda x: x[key], df)))

def k_folds(n_folds, indices):
    """
    Generator for k-fold split of the given indices.

    Parameters
    ----------
    n_folds: int
        Number of folds.
    indices: list
        List of indices to split.

    Yields
    ------
    train_indices: list
        List of indices for the training set.
    test_indices: list
        List of indices for the test set.
    """
    kf = KFold(n_splits=n_folds)
    for train_ix, test_ix in kf.split(indices):
        yield [int(indices[i]) for i in train_ix], [int(indices[i]) for i in test_ix]

def split(X, p):
    """
    Split the given list X into p partitions.

    Parameters
    ----------
    X: list
        The list to split.
    p: int
        The number of partitions.

    Returns
    -------
    partitions: list
        The list of partitions.
    """
    d = len(X) // p
    X_reduced = X[:(d * p)]
    partitions = np.split(np.array(X_reduced), p)
    partitions = [a.tolist() for a in partitions]
    if len(X[d*p:]) != 0:
        partitions += [X[d*p:]]
    return partitions

def pandas_series_to_pytorch(df, device):
    """
    Convert a pandas Series to a PyTorch tensor and move it to a specified device.

    Parameters
    ----------
    df: pandas.Series
        The pandas Series to convert.
    device: torch.device
        The device to move the tensor to.

    Returns
    -------
    torch.Tensor
        The converted tensor moved to the specified device.
    """

    return torch.Tensor(df.to_list()).to(device)

class Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class Decoder(torch.nn.Module):
    def __init__(self, latent_dimension):
        """
        Initialize the decoder.

        Parameters
        ----------
        latent_dimension: int
            The input dimension of the decoder, which should be the same as the output dimension of the encoder.
        """
        super(Decoder, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(latent_dimension, 3),
            torch.nn.GELU(),
            torch.nn.Linear(3,2)
        )

    def forward(self, x):
        o = self.layers(x)
        output = torch.zeros_like(o)
        output[:,0] = torch.sin(o[:,0])
        output[:,1] = torch.cos(o[:,1])
        return output
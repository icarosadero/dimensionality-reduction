import torch
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
import numpy as np
import warnings
from typing import Union, Tuple, Optional

try:
    import pandas as pd
    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False

def train_test_split(
    *arrays,
    test_size: Union[float, int] = 0.25,
    train_size: Optional[Union[float, int]] = None,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[np.ndarray] = None
) -> Tuple:
    """Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas
        dataframes.

    test_size : float or int, default=0.25
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.

    train_size : float or int, optional (default=None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples.

    random_state : int, optional (default=None)
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.

    stratify : array-like, optional (default=None)
        If not None, data is split in a stratified fashion, using this as
        the class labels.

    Returns
    -------
    splitting : list of arrays
        List containing train-test split of inputs.

    Examples
    --------
    >>> import numpy as np
    >>> from mlscratch.utils.data_preprocessing import train_test_split
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 3, 4])
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    """
    
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    n_samples = len(arrays[0])

    if not all(len(X) == n_samples for X in arrays):
        raise ValueError("All arrays must have the same length")

    if test_size is None and train_size is None:
        test_size = 0.25

    if np.asarray(test_size).dtype.kind == 'f':
        if test_size < 0 or test_size > 1:
            raise ValueError(
                'test_size should be float between 0.0 and 1.0 or integer'
            )
    if train_size is not None:
        if np.asarray(train_size).dtype.kind == 'f':
            if train_size < 0 or train_size > 1:
                raise ValueError(
                    'train_size should be float between 0.0 and 1.0 or integer'
                )
            test_size = 1.0 - train_size

    if isinstance(test_size, float):
        n_test = int(test_size * n_samples)
    else:
        n_test = test_size

    if n_test < 0 or n_test > n_samples:
        raise ValueError('test_size cannot be negative or larger than dataset size')

    # Set up random state
    rng = np.random.RandomState(seed=random_state)

    if not shuffle:
        if stratify is not None:
            warnings.warn(
                'Stratified train/test split is not implemented for shuffle=False'
            )
        train_idx = np.arange(n_samples - n_test)
        test_idx = np.arange(n_samples - n_test, n_samples)

    else:
        if stratify is not None:
            unique_labels, label_counts = np.unique(stratify, return_counts=True)
            label_to_idx = {label: np.where(stratify == label)[0] 
                           for label in unique_labels}
            
            train_idx = []
            test_idx = []
            
            for label, count in zip(unique_labels, label_counts):
                label_indices = label_to_idx[label]
                n_label_test = int(test_size * len(label_indices))
                
                shuffled_indices = rng.permutation(len(label_indices))
                label_test_idx = label_indices[shuffled_indices[:n_label_test]]
                label_train_idx = label_indices[shuffled_indices[n_label_test:]]
                
                train_idx.extend(label_train_idx)
                test_idx.extend(label_test_idx)
                
            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)
        else:
            permutation = rng.permutation(n_samples)
            test_idx = permutation[:n_test]
            train_idx = permutation[n_test:]

    result = []
    for X in arrays:
        if PANDAS_INSTALLED and isinstance(X, pd.DataFrame):
            result.extend([X.iloc[train_idx], X.iloc[test_idx]])
        else:
            X = np.asarray(X)
            result.extend([X[train_idx], X[test_idx]])

    return tuple(result)
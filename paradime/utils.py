from datetime import datetime
import torch
import numpy as np
import scipy

from paradime.types import Tensor

def report(message: str) -> None:
    
    now_str = datetime.now().isoformat(
        sep=' ',
        timespec='milliseconds'
    )[:-2]

    print(now_str + ': ' + message)


def _convert_input_to_numpy(
    X: Tensor) -> np.ndarray:
    
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    elif isinstance(X, scipy.sparse.spmatrix):
        X = X.toarray()
    elif isinstance(X, np.ndarray):
        pass
    else:
        raise TypeError(f'Input type {type(X)} not supported')

    return X

def _convert_input_to_torch(
    X: Tensor) -> torch.Tensor:

    if isinstance(X, torch.Tensor):
        pass
    elif isinstance(X, scipy.sparse.spmatrix):
        # TODO: conserve sparseness
        X = torch.tensor(X.toarray())
    elif isinstance(X, np.ndarray):
        X = torch.tensor(X)
    else:
        raise TypeError(f'Input type {type(X)} not supported')

    return X
import platform
if platform.system() != 'Windows':
    import SharedArray as sa
    on_windows = False
else:
    on_windows = True
import numpy as np
import torch
import os

def write_data(data, path):
    assert not on_windows
    if isinstance(data, np.ndarray):
        arr = sa.create(f"shm://{path}", data.shape, dtype=data.dtype)
        arr[:] = data
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
        arr = sa.create(f"shm://{path}", data.shape, dtype=data.dtype)
        arr[:] = data
    else:
        raise NotImplementedError

def read_data(path):
    assert not on_windows
    arr = sa.attach(f"shm://{path}")
    return arr

def read_data_torch(path):
    assert not on_windows
    arr = sa.attach(f"shm://{path}")
    arr = torch.from_numpy(arr)
    return arr

def exists(path):
    assert not on_windows
    return os.path.exists(f'/dev/shm/{path}')

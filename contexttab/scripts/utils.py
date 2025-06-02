import torch
from typing import Dict, Literal, Optional, Tuple, Union



def to_device(x, device: Union[torch.device, int], dtype: Optional[torch.dtype] = None, raise_on_unexpected=True):
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            target_dtype = dtype if v.dtype == torch.float32 else v.dtype
            x[k] = v.to(device, dtype=target_dtype)
        elif isinstance(v, dict):
            x[k] = to_device(v, device, dtype=dtype)
        elif v is not None and raise_on_unexpected:
            raise ValueError(f'Unknown type, {type(v)}')
    return x

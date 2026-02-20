import os
from functools import lru_cache

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def _load_ext():
    """
    JIT-compile and load the C++ extension for 1-bit Seide.

    放在 compressor/onebit_seide_ext.cpp，使用 PyTorch C++ Extension 编译。
    第一次导入时会触发编译，后续复用已编译的模块。
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(this_dir, "onebit_seide_ext.cpp")
    return load(
        name="onebit_seide_ext",
        sources=[src],
        verbose=False,
    )


def onebit_seide_per_column(g: torch.Tensor, col_size: int):
    """
    C++ 版本的 per-column 统计与 bit-pack。
    g: 1D tensor on CPU 或 CUDA
    返回: packed (uint8), a_vec, b_vec
    """
    ext = _load_ext()
    return ext.onebit_seide_per_column(g, int(col_size))


def onebit_seide_reconstruct_from_gathered(
    signs_all: torch.Tensor,
    a_all: torch.Tensor,
    b_all: torch.Tensor,
    numel: int,
    col_size: int,
):
    """
    C++ 版本的重构函数，等价于 Python 版 _onebit_seide_reconstruct_from_gathered。
    """
    ext = _load_ext()
    return ext.onebit_seide_reconstruct_from_gathered(
        signs_all, a_all, b_all, int(numel), int(col_size)
    )


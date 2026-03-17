"""UBER PROTOTYPE!!!"""
# mypy: allow-untyped-defs

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass
from functools import cache
from typing import Any, TYPE_CHECKING
from typing_extensions import TypeVarTuple, Unpack

# from . import _registry

from ... import cutedsl_utils as cu

from ._fa4_impl import (
    _fa4_flash_attention_forward_impl,
    _fa4_flash_attention_backward_impl,
    _fa4_scaled_dot_product_flash_attention_forward_impl,
    _fa4_scaled_dot_product_flash_attention_backward_impl,
)

if TYPE_CHECKING:
    from types import ModuleType

import torch
from torch.library import Library

from torch.nn.attention._registry import (
    current_flash_attention_impl,
    register_flash_attention_impl,
)

def _enable_kernel(dispatch_key, *args, fn, fallback_kernel, **kwargs) -> cu._OpFn:
    # check for FAv4 active, otherwise fallback
    if current_flash_attention_impl() == "FA4":
        return fn(*args, **kwargs)
    else:
        return fallback_kernel(dispatch_key, *args, **kwargs)

def _wrap_kernel(
    op_symbol: str,
    fn,
):
    """
    wrap a given FA4 function to check if FA4 is enabled, otherwise
    fallback to original (previous) implementation
    """
    fallback = torch.library.get_kernel("aten::" + op_symbol, "CUDA")

    wrapped_fn = functools.partial(
        _enable_kernel, fn=fn, fallback_kernel=fallback,
    )
    return wrapped_fn


def _fa4_register_kernels() -> None:
    overrides = {
        "_flash_attention_forward": _fa4_flash_attention_forward_impl,
        "_flash_attention_backward": _fa4_flash_attention_backward_impl,
        "_scaled_dot_product_flash_attention": _fa4_scaled_dot_product_flash_attention_forward_impl,
        "_scaled_dot_product_flash_attention_backward": _fa4_scaled_dot_product_flash_attention_backward_impl,
    }

    for op_symbol, fn in overrides.items():
        cu.register_op_override(
            "aten",
            op_symbol,
            "CUDA",
            _wrap_kernel(op_symbol, fn),
        )

# Register kernels -> dispatcher
_fa4_register_kernels()
# Register implementation -> attention mechanism
register_flash_attention_impl("FA4", register_fn=None)


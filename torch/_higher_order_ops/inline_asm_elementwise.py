# mypy: allow-untyped-defs
import functools
import re

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


__all__ = ["inline_asm_elementwise"]


class InlineAsmElementwiseOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("inline_asm_elementwise")

    def __call__(
        self,
        *inputs: torch.Tensor,
        asm_str: str,
        constraints: str,
        dtype: torch.dtype,
        is_pure: bool = True,
        pack: int = 1,
    ) -> torch.Tensor:
        if not is_pure:
            raise ValueError("inline_asm_elementwise only supports is_pure=True")
        # pyrefly: ignore [missing-attribute]
        return super().__call__(
            *inputs,
            asm_str=asm_str,
            constraints=constraints,
            dtype=dtype,
            is_pure=True,
            pack=pack,
        )


inline_asm_elementwise = InlineAsmElementwiseOp()


def _parse_constraints(constraints: str) -> tuple[int, int]:
    parts = [p.strip() for p in constraints.split(",")]
    n_outputs = sum(1 for p in parts if p.startswith("="))
    n_inputs = len(parts) - n_outputs
    return n_outputs, n_inputs


_DTYPE_TO_CUDA_TYPE = {
    torch.float32: "float",
    torch.float64: "double",
    torch.float16: "__half",
    torch.bfloat16: "__nv_bfloat16",
    torch.int32: "int",
    torch.int64: "long long",
    torch.int16: "short",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
    torch.uint16: "unsigned short",
    torch.uint32: "unsigned int",
    torch.bool: "bool",
}


def _triton_asm_to_cuda_asm(asm_str: str) -> str:
    return re.sub(r"\$(\d+)", r"%\1", asm_str)


@functools.lru_cache(maxsize=256)
def _get_jiterator_fn(
    asm_str: str,
    constraints: str,
    n_inputs: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
):
    from torch.cuda.jiterator import _create_jit_fn

    cuda_asm = _triton_asm_to_cuda_asm(asm_str)

    constraint_parts = [p.strip() for p in constraints.split(",")]
    output_constraints = [p.lstrip("=") for p in constraint_parts if p.startswith("=")]
    input_constraints = [p for p in constraint_parts if not p.startswith("=")]

    if input_dtype not in _DTYPE_TO_CUDA_TYPE:
        raise ValueError(f"Unsupported input dtype for inline asm: {input_dtype}")
    if output_dtype not in _DTYPE_TO_CUDA_TYPE:
        raise ValueError(f"Unsupported output dtype for inline asm: {output_dtype}")

    input_type = _DTYPE_TO_CUDA_TYPE[input_dtype]
    output_type = _DTYPE_TO_CUDA_TYPE[output_dtype]

    input_params = ", ".join(f"{input_type} in{i}" for i in range(n_inputs))
    out_constraints_str = ", ".join(f'"={c}"(result)' for c in output_constraints)
    in_constraints_str = ", ".join(
        f'"{c}"(in{i})' for i, c in enumerate(input_constraints)
    )
    escaped_asm = (
        cuda_asm.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    )

    code = f"""
template <typename T>
{output_type} inline_asm_kernel({input_params}) {{
    {output_type} result;
    asm volatile (
        "{escaped_asm}"
        : {out_constraints_str}
        : {in_constraints_str}
    );
    return result;
}}
"""

    return _create_jit_fn(code)


def _inline_asm_dense(*inputs, asm_str, constraints, dtype, is_pure, pack):
    if not inputs:
        raise ValueError("inline_asm_elementwise requires at least one input tensor")

    inputs = torch.broadcast_tensors(*inputs)

    if not inputs[0].is_cuda:
        raise RuntimeError("inline_asm_elementwise only supports CUDA tensors")

    if pack > 1:
        raise RuntimeError(
            "inline_asm_elementwise with pack > 1 requires torch.compile"
        )

    n_outputs, n_inputs = _parse_constraints(constraints)

    if n_outputs != 1:
        raise ValueError(f"Expected 1 output constraint, got {n_outputs}")

    if n_inputs != len(inputs):
        raise ValueError(
            f"Constraint string specifies {n_inputs} inputs but got "
            f"{len(inputs)} tensor(s)"
        )

    jit_fn = _get_jiterator_fn(
        asm_str=asm_str,
        constraints=constraints,
        n_inputs=len(inputs),
        input_dtype=inputs[0].dtype,
        output_dtype=dtype,
    )

    return jit_fn(*inputs)


@inline_asm_elementwise.py_impl(DispatchKey.CompositeExplicitAutograd)
def inline_asm_eager(*inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    return _inline_asm_dense(
        *inputs,
        asm_str=asm_str,
        constraints=constraints,
        dtype=dtype,
        is_pure=is_pure,
        pack=pack,
    )


inline_asm_elementwise.py_autograd_impl(
    autograd_not_implemented(inline_asm_elementwise, deferred_error=True)
)


@inline_asm_elementwise.py_impl(FakeTensorMode)
def inline_asm_fake(mode, *inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    with mode:
        broadcasted = torch.broadcast_tensors(*inputs)
        return torch.empty_like(broadcasted[0], dtype=dtype)


@inline_asm_elementwise.py_impl(DispatchKey.Meta)
def inline_asm_meta(*inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    broadcasted = torch.broadcast_tensors(*inputs)
    return torch.empty_like(broadcasted[0], dtype=dtype)


def trace_inline_asm(
    proxy_mode, func_overload, *inputs, asm_str, constraints, dtype, is_pure, pack
):
    with disable_proxy_modes_tracing():
        broadcasted = torch.broadcast_tensors(*inputs)
        out = torch.empty_like(broadcasted[0], dtype=dtype)

    node_args = inputs
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        func_overload,
        proxy_args,
        {
            "asm_str": asm_str,
            "constraints": constraints,
            "dtype": dtype,
            "is_pure": is_pure,
            "pack": pack,
        },
        name="inline_asm_elementwise",
    )

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@inline_asm_elementwise.py_impl(ProxyTorchDispatchMode)
def inline_asm_proxy(mode, *inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    return trace_inline_asm(
        mode,
        inline_asm_elementwise,
        *inputs,
        asm_str=asm_str,
        constraints=constraints,
        dtype=dtype,
        is_pure=is_pure,
        pack=pack,
    )


@inline_asm_elementwise.py_functionalize_impl
def inline_asm_func(ctx, *inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    unwrapped_inputs = ctx.unwrap_tensors(inputs)

    with ctx.redispatch_to_next():
        res = inline_asm_elementwise(
            *unwrapped_inputs,
            asm_str=asm_str,
            constraints=constraints,
            dtype=dtype,
            pack=pack,
        )
    return ctx.wrap_tensors(res)

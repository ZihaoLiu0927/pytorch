import torch

from ... import triton_utils as tu


def triton_to_mxfp8_dim0(
    keyset: torch.DispatchKeySet,
    x: torch.Tensor,
    inner_block_size: int = 32,
    scaling_mode: str = "rceil",
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.backends.triton.enabled:
        raise RuntimeError(
            "Triton native ops are disabled (torch.backends.triton.enabled = False)"
        )
    import triton

    from .mxfp8_triton_kernel import to_mxfp8_dim0_kernel

    """
    Input:
    * `x` - input tensor, in row major memory layout
    * `inner_block_size` - size of tiles to scale across, default is 32 for MX recipes
    * `scaling_mode` - floor or rceil

    Output:
    * `output`: the `float8_e4m3fn` values of `x` cast to mxfp8 across dim0 (rowwise)
    * `scale`: the `e8m0` values of `x_scale` used to cast `x` to mxfp8 across dim0
    """
    assert x.is_contiguous(), "`x` must be contiguous"
    assert inner_block_size <= 32, "inner_block_size must be <= 32"
    assert x.dtype == torch.bfloat16, (
        f"only bfloat16 inputs are supported, got {x.dtype}"
    )
    assert scaling_mode in ("floor", "rceil"), (
        "only floor and rceil scaling modes are supported"
    )

    # Reshape tensor to 2d if necessary and get shape
    x_orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    n_rows, n_cols = x.shape

    assert n_cols % inner_block_size == 0, (
        "columns must be divisible by inner block size"
    )

    # Create output tensors
    output = torch.empty((n_rows, n_cols), dtype=torch.float8_e4m3fn, device=x.device)

    # Create scale tensors for rowwise scaling
    scale = torch.empty(
        (n_rows, n_cols // inner_block_size),
        dtype=torch.uint8,
        device=x.device,
    )

    # Calculate grid dimensions based on tile size
    grid = lambda META: (
        triton.cdiv(n_rows, META["ROW_TILE_SIZE"]),
        triton.cdiv(n_cols, META["COL_TILE_SIZE"]),
    )

    # Launch the kernel
    # torch.library.wrap_triton(to_mxfp8_dim0_kernel)[grid](
    to_mxfp8_dim0_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        scale_ptr=scale,
        n_rows=n_rows,
        n_cols=n_cols,
        SCALE_BLOCK_SIZE=inner_block_size,
        SCALING_MODE=scaling_mode.lower(),
    )

    # Reshape output back to original shape
    output = output.reshape(x_orig_shape)
    scale = scale.reshape(*x_orig_shape[:-1], scale.shape[-1])

    return (
        output,
        scale.view(torch.float8_e8m0fnu),
    )


# torch.library.register_autograd(
#     "mxfp8_quant::triton_to_mxfp8_dim0",
#     None,
#     setup_context=None,
# )


# @torch.library.register_fake("mxfp8_quant::triton_to_mxfp8_dim0")
def _mxfp8_quant_fake(x, inner_block_size, scaling_mode):
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn)
    sf = torch.empty(
        (x.shape // 128, x.shape // inner_block_size), dtype=torch.float8_e8m0fnu
    )

    return out, sf


def register_to_dispatcher():
    # NOTE: This doesn't actually work..
    #       Want to use a custom op (unsupported right now)
    tu.register_op_override(
        "mxfp8_quant",
        "triton_to_mxfp8_dim0",
        "CUDA",
        triton_to_mxfp8_dim0,
    )


register_to_dispatcher()

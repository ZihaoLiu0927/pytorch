import triton
import triton.language as tl


@triton.jit
def _triton_calculate_scale(x, axis, SCALING_MODE: tl.constexpr):
    # There is no good support for accessing globals from a jit'ed triton
    # function, so we redefine them here. Since this is prototype code which
    # we plan to remove after torch.compile catches up, this is fine.
    target_max_pow2 = 8
    e8m0_exponent_bias = 127
    bf16_mbits = 7
    bf16_exp_bias = 127
    fp32_mbits = 23

    # Find the maximum absolute value for each row
    max_abs = tl.max(x, axis=axis)

    # Compute e8m0 biased scale using either RCEIL or FLOOR rounding.
    if SCALING_MODE == "rceil":
        # RCEIL scaling mode using PTX instruction supported on sm100.
        # The input should be: amax / 448.0
        # where 448.0 is the max representable value in FP8 E4M3 format.
        F8E4M3_MAX_RCP: tl.constexpr = 1.0 / 448.0
        scale_input = max_abs.to(tl.float32) * F8E4M3_MAX_RCP

        # The PTX instruction outputs a packed uint16 where:
        # - high byte = E8M0 of first input (0.0 in our case)
        # - low byte = E8M0 of second input (scale_input)
        # Casting uint16 to uint8 naturally truncates to the low byte.
        scale_e8m0_biased = tl.inline_asm_elementwise(
            asm="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
            constraints="=h,r",
            args=[scale_input.to(tl.float32, bitcast=False)],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        ).to(tl.uint8)
    else:
        tl.static_assert(SCALING_MODE == "floor")

        # Original floor implementation
        # Calculate the e8m0 scale by extracting the exponent (floor)
        max_abs = max_abs.to(tl.bfloat16)
        max_abs_int16 = max_abs.to(tl.int16, bitcast=True)
        extracted_pow2 = ((max_abs_int16 >> bf16_mbits) & 0b11111111) - bf16_exp_bias
        extracted_pow2 = extracted_pow2 - target_max_pow2
        scale_e8m0_unbiased = extracted_pow2.to(tl.bfloat16)

        # Clamp to exponents that can be represented in e8m0
        # Add 1 to capture NaNs
        scale_e8m0_unbiased = tl.clamp(
            scale_e8m0_unbiased, -1 * e8m0_exponent_bias, e8m0_exponent_bias + 1
        )

        # Create the biased e8m0 representation and cast it to 8 bits
        scale_e8m0_biased = scale_e8m0_unbiased + e8m0_exponent_bias
        scale_e8m0_biased = scale_e8m0_biased.to(tl.uint8)

    # TODO(future PR): add NaN handling here,
    # https://github.com/pytorch/pytorch/pull/100572 will likely be useful to
    # get proper NaN propagation working
    # Calculate the scale in floating point.
    scale_fp = (scale_e8m0_biased.to(tl.int32) << fp32_mbits).to(
        tl.float32, bitcast=True
    )

    fp32_exp_bias = 127.0
    fp32_min_normal = tl.exp2(-fp32_exp_bias + 1)
    scale_fp = tl.clamp(scale_fp, min=fp32_min_normal, max=float("inf"))

    return scale_fp, scale_e8m0_biased


def _get_mxfp8_quant_autotune_configs():
    # Values to sweep over here were determined by a manual
    # sweep over a small set of shapes, it's likely that this
    # can be improved in the future.
    results = []
    for ROW_TILE_SIZE in (128, 256, 512):
        # TODO: we can't use 512 for COL_TILE_SIZE.
        # This is likely a triton bug, tracked in
        # https://github.com/pytorch/ao/issues/3362
        for COL_TILE_SIZE in (128, 256):
            for num_warps in (4, 8):
                for num_stages in (2, 3):
                    config = triton.Config(
                        {
                            "ROW_TILE_SIZE": ROW_TILE_SIZE,
                            "COL_TILE_SIZE": COL_TILE_SIZE,
                        },
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                    results.append(config)
    return results


@triton.autotune(
    configs=_get_mxfp8_quant_autotune_configs(),
    key=["n_cols", "SCALE_BLOCK_SIZE"],
)
@triton.jit
def to_mxfp8_dim0_kernel(
    x_ptr,
    output_ptr,
    scale_ptr,
    n_rows,
    n_cols,
    ROW_TILE_SIZE: tl.constexpr,
    COL_TILE_SIZE: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,  # should be 32 for MX
    SCALING_MODE: tl.constexpr,
):
    """
    Quantizes a high precision tensor to mxfp8 rowwise (1x32 scaling granularity).
    """

    SCALE_BLOCKS_PER_COL_TILE: tl.constexpr = COL_TILE_SIZE // SCALE_BLOCK_SIZE

    # Get program ID
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    start_row = pid_row * ROW_TILE_SIZE
    start_col = pid_col * COL_TILE_SIZE
    row_offs = start_row + tl.arange(0, ROW_TILE_SIZE)[:, None]
    col_offs = start_col + tl.arange(0, COL_TILE_SIZE)[None, :]

    # Compute memory offsets for row-major layout (rows, cols)
    row_major_offsets = (row_offs * n_cols + col_offs).to(tl.int32)

    # Load the entire block in a single operation
    # shape: (ROW_TILE_SIZE, COL_TILE_SIZE)
    mask = (row_offs < n_rows) & (col_offs < n_cols)
    x_block = tl.load(x_ptr + row_major_offsets, mask=mask)

    # Reshape to inner tile size for rowwise scaling
    # shape: (ROW_TILE_SIZE, COL_TILE_SIZE) -> (ROW_TILE_SIZE * BLOCKS_PER_COL_TILE, SCALE_BLOCK_SIZE)
    x_block_r = x_block.reshape(
        ROW_TILE_SIZE * SCALE_BLOCKS_PER_COL_TILE, SCALE_BLOCK_SIZE
    )

    # Calculate the absolute values of elements in the block
    x_block_abs_r = tl.abs(x_block_r)

    # Find the maximum absolute value for each row (across columns)
    # shape: (ROW_TILE_SIZE * BLOCKS_PER_COL_TILE,)
    scale_fp32_r, scale_e8m0_r = _triton_calculate_scale(
        x_block_abs_r, axis=1, SCALING_MODE=SCALING_MODE
    )

    # Divide each row by scale
    # Broadcasting scale to match x_block's shape
    # x_block_r shape:
    #    (ROW_TILE_SIZE * BLOCKS_PER_COL_TILE, SCALE_BLOCK_SIZE)
    # scale[:, None] shape:
    #    (ROW_TILE_SIZE * BLOCKS_PER_COL_TILE, 1)
    scaled_data_r = x_block_r / scale_fp32_r[:, None]

    # Reshape back to original tile size
    e4m3_data_2d = tl.reshape(scaled_data_r, ROW_TILE_SIZE, COL_TILE_SIZE).to(
        tl.float8e4nv
    )

    # Store the row-normalized result in row-major format
    tl.store(output_ptr + row_major_offsets, e4m3_data_2d, mask=mask)

    # Calculate scale offsets to write to
    scales_per_row = n_cols // SCALE_BLOCK_SIZE
    scale_row_indices = pid_row * ROW_TILE_SIZE + tl.arange(0, ROW_TILE_SIZE)[:, None]
    scale_col_indices = (
        pid_col * SCALE_BLOCKS_PER_COL_TILE
        + tl.arange(0, SCALE_BLOCKS_PER_COL_TILE)[None, :]
    )
    scale_offsets = scale_row_indices * scales_per_row + scale_col_indices

    # Store e8m0 scales
    scale_mask = (scale_row_indices < n_rows) & (scale_col_indices < scales_per_row)
    scale_e8m0_2d = scale_e8m0_r.reshape(ROW_TILE_SIZE, SCALE_BLOCKS_PER_COL_TILE)
    tl.store(scale_ptr + scale_offsets, scale_e8m0_2d, mask=scale_mask)

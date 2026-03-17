# Owner(s): ["module: higher order operators"]
"""
Tests for inline_asm_elementwise higher-order operator.

Tests verify:
1. Bitwise equivalence between eager (Jiterator) and compiled (Inductor) paths
2. Correctness via approximate comparison with reference PyTorch ops
"""

import unittest
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch.testing._internal.common_cuda import SM70OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUDA,
    TestCase,
)


@dataclass
class AsmTestCase:
    name: str
    input_gen_fn: Callable
    asm_str: str
    constraints: str
    dtype: torch.dtype
    approx_fn: Callable
    pack: int = 1
    compile_only: bool = False
    min_sm: int = 70


TEST_CASES = [
    # Basic float32 operations
    AsmTestCase(
        "identity_f32",
        lambda: (torch.randn(100, device="cuda", dtype=torch.float32),),
        "mov.f32 $0, $1;",
        "=f,f",
        torch.float32,
        lambda x: x,
    ),
    AsmTestCase(
        "add_f32",
        lambda: (
            torch.randn(100, device="cuda", dtype=torch.float32),
            torch.randn(100, device="cuda", dtype=torch.float32),
        ),
        "add.f32 $0, $1, $2;",
        "=f,f,f",
        torch.float32,
        lambda x, y: x + y,
    ),
    AsmTestCase(
        "mul_f32",
        lambda: (
            torch.randn(100, device="cuda", dtype=torch.float32),
            torch.randn(100, device="cuda", dtype=torch.float32),
        ),
        "mul.f32 $0, $1, $2;",
        "=f,f,f",
        torch.float32,
        lambda x, y: x * y,
    ),
    AsmTestCase(
        "fma_f32",
        lambda: (
            torch.randn(100, device="cuda", dtype=torch.float32),
            torch.randn(100, device="cuda", dtype=torch.float32),
            torch.randn(100, device="cuda", dtype=torch.float32),
        ),
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        torch.float32,
        lambda a, b, c: a * b + c,
    ),
    # Multi-line PTX with curly braces
    AsmTestCase(
        "double_multiline",
        lambda: (torch.randn(100, device="cuda", dtype=torch.float32),),
        "{.reg .f32 tmp; mov.f32 tmp, $1; add.f32 $0, tmp, tmp;}",
        "=f,f",
        torch.float32,
        lambda x: x * 2,
    ),
    # bf16/fp16 upcasting (compile-only: Jiterator can't handle dtype mismatch)
    AsmTestCase(
        "bf16_upcast",
        lambda: (torch.randn(100, device="cuda", dtype=torch.bfloat16),),
        "add.f32 $0, $1, $1;",
        "=f,f",
        torch.float32,
        lambda x: x.float() * 2,
        compile_only=True,
    ),
    AsmTestCase(
        "fp16_upcast",
        lambda: (torch.randn(100, device="cuda", dtype=torch.float16),),
        "add.f32 $0, $1, $1;",
        "=f,f",
        torch.float32,
        lambda x: x.float() * 2,
        compile_only=True,
    ),
    # Integer operations
    AsmTestCase(
        "bitwise_and",
        lambda: (
            torch.randint(0, 2**16, (100,), device="cuda", dtype=torch.int32),
            torch.randint(0, 2**16, (100,), device="cuda", dtype=torch.int32),
        ),
        "and.b32 $0, $1, $2;",
        "=r,r,r",
        torch.int32,
        lambda x, y: x & y,
    ),
    AsmTestCase(
        "bitwise_or",
        lambda: (
            torch.randint(0, 2**16, (100,), device="cuda", dtype=torch.int32),
            torch.randint(0, 2**16, (100,), device="cuda", dtype=torch.int32),
        ),
        "or.b32 $0, $1, $2;",
        "=r,r,r",
        torch.int32,
        lambda x, y: x | y,
    ),
    # Output dtype differs from input (compile-only: Jiterator returns input dtype)
    AsmTestCase(
        "exponent_extract",
        lambda: (
            torch.tensor([1.0, 2.0, 0.5, 16.0], device="cuda", dtype=torch.float32),
        ),
        "{.reg .b32 t; mov.b32 t,$1; shr.u32 t,t,23; and.b32 $0,t,0xFF;}",
        "=r,f",
        torch.int32,
        lambda x: ((x.view(torch.int32) >> 23) & 0xFF).to(torch.int32),
        compile_only=True,
    ),
    # Mixed constraint types: "r" input, "h" output (compile-only)
    AsmTestCase(
        "truncate_to_uint16",
        lambda: (torch.randint(0, 256, (100,), device="cuda", dtype=torch.int32),),
        "cvt.u16.u32 $0, $1;",
        "=h,r",
        torch.uint16,
        lambda x: x.to(torch.uint16),
        compile_only=True,
    ),
    # Broadcasting
    AsmTestCase(
        "broadcast_add",
        lambda: (
            torch.randn(4, 1, device="cuda", dtype=torch.float32),
            torch.randn(1, 8, device="cuda", dtype=torch.float32),
        ),
        "add.f32 $0, $1, $2;",
        "=f,f,f",
        torch.float32,
        lambda x, y: x + y,
    ),
    # Non-contiguous
    AsmTestCase(
        "noncontiguous",
        lambda: (torch.randn(8, 16, device="cuda", dtype=torch.float32).t(),),
        "mov.f32 $0, $1;",
        "=f,f",
        torch.float32,
        lambda x: x,
    ),
    # fp16/bf16 native asm (compile-only: inductor computes in fp32, needs downcast)
    AsmTestCase(
        "add_fp16_native",
        lambda: (
            torch.randn(100, device="cuda", dtype=torch.float16),
            torch.randn(100, device="cuda", dtype=torch.float16),
        ),
        "add.f16 $0, $1, $2;",
        "=h,h,h",
        torch.float16,
        lambda x, y: x + y,
        compile_only=True,
    ),
    AsmTestCase(
        "add_bf16_native",
        lambda: (
            torch.randn(100, device="cuda", dtype=torch.bfloat16),
            torch.randn(100, device="cuda", dtype=torch.bfloat16),
        ),
        "add.bf16 $0, $1, $2;",
        "=h,h,h",
        torch.bfloat16,
        lambda x, y: x + y,
        compile_only=True,
        min_sm=90,
    ),
    # pack=2: each asm invocation processes 2 elements (compile-only)
    AsmTestCase(
        "identity_pack2",
        lambda: (torch.randn(128, device="cuda", dtype=torch.float32),),
        "mov.b32 $0, $2; mov.b32 $1, $3;",
        "=r,=r,r,r",
        torch.float32,
        lambda x: x,
        pack=2,
        compile_only=True,
    ),
    AsmTestCase(
        "add_pack2",
        lambda: (
            torch.randn(128, device="cuda", dtype=torch.float32),
            torch.randn(128, device="cuda", dtype=torch.float32),
        ),
        "add.f32 $0, $2, $4; add.f32 $1, $3, $5;",
        "=f,=f,f,f,f,f",
        torch.float32,
        lambda x, y: x + y,
        pack=2,
        compile_only=True,
    ),
]
TEST_CASE_NAMES = [tc.name for tc in TEST_CASES]


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@unittest.skipIf(not SM70OrLater, "Requires SM70+")
@instantiate_parametrized_tests
class TestInlineAsmElementwise(TestCase):
    """Parametrized tests for inline_asm_elementwise."""

    @parametrize(
        "case_idx", list(range(len(TEST_CASES))), name_fn=lambda i: TEST_CASE_NAMES[i]
    )
    def test_eager_vs_compiled_bitwise(self, case_idx):
        """Verify eager and compiled produce bitwise identical results."""
        tc = TEST_CASES[case_idx]
        if torch.cuda.get_device_capability() < (tc.min_sm // 10, tc.min_sm % 10):
            self.skipTest(f"Requires SM{tc.min_sm}+")
        inputs = tc.input_gen_fn()

        def fn(*args):
            return inline_asm_elementwise(
                *args,
                asm_str=tc.asm_str,
                constraints=tc.constraints,
                dtype=tc.dtype,
                pack=tc.pack,
            )

        torch._dynamo.reset()
        compiled_result = torch.compile(fn, backend="inductor")(*inputs)

        if tc.compile_only:
            expected = tc.approx_fn(*inputs)
            self.assertTrue(
                torch.allclose(
                    compiled_result.float(), expected.float(), rtol=1e-5, atol=1e-5
                ),
                f"Compiled differs from expected for {tc.name}:\n"
                f"  max diff: {(compiled_result.float() - expected.float()).abs().max()}",
            )
        else:
            eager_result = fn(*inputs)
            self.assertTrue(
                torch.equal(eager_result, compiled_result),
                f"Eager and compiled differ for {tc.name}:\n"
                f"  max diff: {(eager_result.float() - compiled_result.float()).abs().max()}",
            )

    @parametrize(
        "case_idx", list(range(len(TEST_CASES))), name_fn=lambda i: TEST_CASE_NAMES[i]
    )
    def test_correctness(self, case_idx):
        """Verify result matches reference function."""
        tc = TEST_CASES[case_idx]
        if torch.cuda.get_device_capability() < (tc.min_sm // 10, tc.min_sm % 10):
            self.skipTest(f"Requires SM{tc.min_sm}+")
        inputs = tc.input_gen_fn()

        def fn(*args):
            return inline_asm_elementwise(
                *args,
                asm_str=tc.asm_str,
                constraints=tc.constraints,
                dtype=tc.dtype,
                pack=tc.pack,
            )

        if tc.compile_only:
            torch._dynamo.reset()
            result = torch.compile(fn, backend="inductor")(*inputs)
        else:
            result = fn(*inputs)
        expected = tc.approx_fn(*inputs)

        self.assertTrue(
            torch.allclose(result.float(), expected.float(), rtol=1e-5, atol=1e-5),
            f"Result differs from expected for {tc.name}:\n"
            f"  max diff: {(result.float() - expected.float()).abs().max()}",
        )


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
class TestInlineAsmElementwiseErrors(TestCase):
    """Tests for error handling."""

    def test_error_no_inputs(self):
        with self.assertRaises(ValueError):
            inline_asm_elementwise(
                asm_str="mov.f32 $0, 1.0;",
                constraints="=f",
                dtype=torch.float32,
            )

    def test_error_constraint_mismatch(self):
        x = torch.randn(100, device="cuda", dtype=torch.float32)
        y = torch.randn(100, device="cuda", dtype=torch.float32)
        with self.assertRaises(ValueError):
            inline_asm_elementwise(
                x,
                y,
                asm_str="add.f32 $0, $1, $2;",
                constraints="=f,f",
                dtype=torch.float32,
            )

    def test_error_cpu_tensor(self):
        x = torch.randn(100, dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            inline_asm_elementwise(
                x,
                asm_str="mov.f32 $0, $1;",
                constraints="=f,f",
                dtype=torch.float32,
            )


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@unittest.skipIf(not SM70OrLater, "Requires SM70+")
class TestInlineAsmElementwiseEdgeCases(TestCase):
    """Tests for edge cases."""

    def test_empty_tensor(self):
        x = torch.empty(0, device="cuda", dtype=torch.float32)
        result = inline_asm_elementwise(
            x, asm_str="mov.f32 $0, $1;", constraints="=f,f", dtype=torch.float32
        )
        self.assertEqual(result.shape, torch.Size([0]))

    def test_scalar_tensor(self):
        x = torch.tensor(3.14, device="cuda", dtype=torch.float32)
        result = inline_asm_elementwise(
            x, asm_str="mov.f32 $0, $1;", constraints="=f,f", dtype=torch.float32
        )
        self.assertEqual(result.shape, torch.Size([]))
        self.assertTrue(torch.allclose(result, x))

    def test_4d_tensor(self):
        x = torch.randn(2, 3, 4, 5, device="cuda", dtype=torch.float32)
        result = inline_asm_elementwise(
            x, asm_str="mov.f32 $0, $1;", constraints="=f,f", dtype=torch.float32
        )
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.equal(result, x))

    def test_composition_with_pytorch_ops(self):
        def fn(x, y):
            z = x * 2
            w = inline_asm_elementwise(
                z,
                y,
                asm_str="add.f32 $0, $1, $2;",
                constraints="=f,f,f",
                dtype=torch.float32,
            )
            return w + 1.0

        x = torch.randn(100, device="cuda", dtype=torch.float32)
        y = torch.randn(100, device="cuda", dtype=torch.float32)

        eager_result = fn(x, y)
        compiled_fn = torch.compile(fn, backend="inductor")
        compiled_result = compiled_fn(x, y)

        self.assertTrue(torch.equal(eager_result, compiled_result))
        self.assertTrue(torch.allclose(eager_result, x * 2 + y + 1.0))

    def test_dynamic_shapes(self):
        def fn(x, y):
            return inline_asm_elementwise(
                x,
                y,
                asm_str="add.f32 $0, $1, $2;",
                constraints="=f,f,f",
                dtype=torch.float32,
            )

        compiled_fn = torch.compile(fn, backend="inductor", dynamic=True)

        for size in [50, 100, 200]:
            x = torch.randn(size, device="cuda", dtype=torch.float32)
            y = torch.randn(size, device="cuda", dtype=torch.float32)
            eager_result = fn(x, y)
            compiled_result = compiled_fn(x, y)
            self.assertTrue(torch.equal(eager_result, compiled_result))


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@unittest.skipIf(not SM70OrLater, "Requires SM70+")
class TestInlineAsmPackPadding(TestCase):
    """Test that pack padding works when block size < pack."""

    def test_pack2_xblock1_padding(self):
        """Force XBLOCK=1 with pack=2 so padding is needed."""
        from torch._inductor.choices import InductorChoices
        from torch._inductor.codegen.triton import FixedTritonConfig
        from torch._inductor.utils import run_and_get_code
        from torch.testing import FileCheck

        class ForceXBlock1(InductorChoices):
            def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
                return {
                    **kernel_kwargs,
                    "fixed_config": FixedTritonConfig({"XBLOCK": 1}),
                }

        def fn(x):
            return inline_asm_elementwise(
                x,
                asm_str="mov.b32 $0, $2; mov.b32 $1, $3;",
                constraints="=r,=r,r,r",
                dtype=torch.float32,
                pack=2,
            )

        x = torch.randn(128, device="cuda", dtype=torch.float32)
        with torch._inductor.virtualized.V.set_choices_handler(ForceXBlock1()):
            torch._dynamo.reset()
            result, (code,) = run_and_get_code(torch.compile(fn, backend="inductor"), x)

        self.assertTrue(torch.equal(result, x))
        # Verify padding helpers are emitted in the generated code
        FileCheck().check("inline_asm_pack").check("inline_asm_unpack").run(code)

    def test_pack4_xblock1_padding(self):
        """Force XBLOCK=1 with pack=4 so padding is needed."""
        from torch._inductor.choices import InductorChoices
        from torch._inductor.codegen.triton import FixedTritonConfig
        from torch._inductor.utils import run_and_get_code
        from torch.testing import FileCheck

        class ForceXBlock1(InductorChoices):
            def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
                return {
                    **kernel_kwargs,
                    "fixed_config": FixedTritonConfig({"XBLOCK": 1}),
                }

        def fn(x):
            return inline_asm_elementwise(
                x,
                asm_str="mov.b32 $0, $4; mov.b32 $1, $5; mov.b32 $2, $6; mov.b32 $3, $7;",
                constraints="=r,=r,=r,=r,r,r,r,r",
                dtype=torch.float32,
                pack=4,
            )

        x = torch.randn(128, device="cuda", dtype=torch.float32)
        with torch._inductor.virtualized.V.set_choices_handler(ForceXBlock1()):
            torch._dynamo.reset()
            result, (code,) = run_and_get_code(torch.compile(fn, backend="inductor"), x)

        self.assertTrue(torch.equal(result, x))
        FileCheck().check("inline_asm_pack").check("inline_asm_unpack").run(code)


if __name__ == "__main__":
    run_tests()

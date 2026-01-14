# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.pipelining._utils import _DTensorMeta
from torch.distributed.pipelining.schedules import Schedule1F1B
from torch.distributed.pipelining.stage import PipelineStage
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
    SequenceParallel,
)
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_accelerator_dist_backend,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_MULTIACCELERATOR,
)

if TYPE_CHECKING:
    from collections.abc import Callable


d_hid = 256
batch_size = 64
n_microbatches = 4

MODEL_SEED = 0
INPUT_SEED = 42
TARGET_SEED = 123


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = dist.get_default_backend_for_device(device_type)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x
            * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
            * self.weight
        )


class NormMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


class NormMLPStack(nn.Module):
    def __init__(self, dim: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([NormMLP(dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def apply_tp_replicate(block: NormMLP, tp_mesh: DeviceMesh) -> None:
    parallelize_module(
        block,
        tp_mesh,
        {
            "fc1": ColwiseParallel(input_layouts=Replicate(), use_local_output=False),
            "fc2": RowwiseParallel(output_layouts=Replicate(), use_local_output=False),
        },
    )
    for name, param in block.norm.named_parameters():
        block.norm.register_parameter(
            name,
            nn.Parameter(
                DTensor.from_local(param, tp_mesh, [Replicate()], run_check=False)
            ),
        )


def apply_tp_shard(block: NormMLP, tp_mesh: DeviceMesh) -> None:
    parallelize_module(
        block,
        tp_mesh,
        {
            "norm": SequenceParallel(use_local_output=False),
            "fc1": ColwiseParallel(
                input_layouts=Shard(1),
                use_local_output=False,  # type: ignore[arg-type]
            ),
            "fc2": RowwiseParallel(
                output_layouts=Shard(1),
                use_local_output=False,  # type: ignore[arg-type]
            ),
        },
    )


def _loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (output - target).pow(2).mean()


class DTensorPPIntegrationBase(MultiProcContinuousTest):
    world_size = 4

    @classmethod
    def backend_str(cls) -> str:
        return backend

    @classmethod
    def device_type(cls) -> str:
        return device_type

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def init_pg(self) -> None:
        if device_type == "cuda":
            torch.cuda.set_device(self.device)

    def _make_mesh(self) -> DeviceMesh:
        return init_device_mesh(device_type, (2, 2), mesh_dim_names=("pp", "tp"))

    def _build_baseline_and_clones(
        self,
        n_layers: int,
        tp_mesh: DeviceMesh,
        apply_tp: Callable,
    ) -> tuple[NormMLPStack, NormMLPStack]:
        torch.manual_seed(MODEL_SEED)
        baseline = NormMLPStack(d_hid, n_layers)

        ref_model = copy.deepcopy(baseline).to(self.device)
        pp_model = copy.deepcopy(baseline).to(self.device)

        for layer in ref_model.layers:
            apply_tp(layer, tp_mesh)
        for layer in pp_model.layers:
            apply_tp(layer, tp_mesh)

        return ref_model, pp_model

    def _make_full_batch_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(INPUT_SEED)
        input_full = torch.randn(batch_size, d_hid, device="cpu").to(self.device)
        torch.manual_seed(TARGET_SEED)
        target_full = torch.randn(batch_size, d_hid, device="cpu").to(self.device)
        return input_full, target_full

    def _wrap_full_tensor_as_dtensor(
        self,
        tensor: torch.Tensor,
        tp_mesh: DeviceMesh,
        placements: list,
    ) -> DTensor:
        tp_rank = tp_mesh.get_local_rank()
        tp_size = tp_mesh.size()

        if len(placements) != 1:
            raise ValueError(f"Expected single placement, got {placements}")

        placement = placements[0]
        if isinstance(placement, Replicate):
            local = tensor
        elif isinstance(placement, Shard):
            local = tensor.chunk(tp_size, dim=placement.dim)[tp_rank].contiguous()
        else:
            raise ValueError(f"Unsupported placement: {placement}")

        return DTensor.from_local(
            local,
            tp_mesh,
            placements,
            shape=tensor.shape,
            stride=tensor.stride(),
            run_check=False,
        )

    def _make_presharded_input_dtensor(
        self,
        input_full: torch.Tensor,
        tp_mesh: DeviceMesh,
    ) -> DTensor:
        tp_rank = tp_mesh.get_local_rank()
        local_shard = input_full.chunk(tp_mesh.size(), dim=1)[tp_rank].contiguous()
        return DTensor.from_local(
            local_shard,
            tp_mesh,
            [Shard(1)],
            shape=input_full.shape,
            stride=input_full.stride(),
            run_check=False,
        )

    def _run_reference_microbatched(
        self,
        ref_model: NormMLPStack,
        input_dt: DTensor,
        target_dt: DTensor,
    ) -> tuple[
        dict[str, torch.Tensor | None],
        dict[int, tuple[DTensor, DTensor, DTensor | None, DTensor | None]],
    ]:
        ref_model.zero_grad(set_to_none=True)

        input_chunks = torch.tensor_split(input_dt, n_microbatches)
        target_chunks = torch.tensor_split(target_dt, n_microbatches)

        captured_io: dict[int, tuple[DTensor, DTensor]] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []

        for stage_idx, stage_mod in enumerate(ref_model.layers):

            def _capture_io(
                _module: nn.Module,
                args: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                idx: int = stage_idx,
            ) -> None:
                stage_input = cast(DTensor, args[0])
                stage_output = cast(DTensor, output)
                stage_input.retain_grad()
                stage_output.retain_grad()
                captured_io[idx] = (stage_input, stage_output)

            hooks.append(stage_mod.register_forward_hook(_capture_io))

        static_stage_meta: dict[
            int, tuple[DTensor, DTensor, DTensor | None, DTensor | None]
        ] = {}

        for mb_idx, (input_chunk, target_chunk) in enumerate(
            zip(input_chunks, target_chunks)
        ):
            output = ref_model(input_chunk)
            loss = _loss_fn(output, target_chunk)
            loss.backward()

            if mb_idx == 0:
                for stage_idx in range(len(ref_model.layers)):
                    stage_input, stage_output = captured_io[stage_idx]
                    input_args = self._empty_dt_from(stage_input, True)
                    output_args = self._empty_dt_from(
                        stage_output, stage_output.requires_grad
                    )
                    input_grads = (
                        self._empty_dt_from(cast(DTensor, stage_input.grad), False)
                        if stage_input.grad is not None
                        else None
                    )
                    output_grads = (
                        self._empty_dt_from(cast(DTensor, stage_output.grad), False)
                        if stage_output.grad is not None
                        else None
                    )

                    if stage_idx == 0:
                        input_grads = None
                    if stage_idx == (len(ref_model.layers) - 1):
                        output_grads = None

                    static_stage_meta[stage_idx] = (
                        input_args,
                        output_args,
                        input_grads,
                        output_grads,
                    )

                for hook in hooks:
                    hook.remove()
                hooks = []

        for param in ref_model.parameters():
            if param.grad is not None:
                param.grad.div_(n_microbatches)

        ref_grads = {
            name: grad.clone() if grad is not None else grad
            for name, grad in (
                (param_name, param.grad)
                for param_name, param in ref_model.named_parameters()
            )
        }
        return ref_grads, static_stage_meta

    def _empty_dt_from(self, dt: DTensor, requires_grad: bool) -> DTensor:
        meta = _DTensorMeta.from_dtensor(dt)
        empty_local = torch.empty(meta.shape, dtype=meta.dtype, device=self.device)
        result = DTensor.from_local(
            empty_local,
            dt.device_mesh,
            list(meta.placements),
            shape=meta.global_shape,
            stride=meta.global_stride,
            run_check=False,
        )
        if requires_grad:
            result.requires_grad_(True)
        return result

    def _make_stage_for_pp_run(
        self,
        pp_model: NormMLPStack,
        stage_index: int,
        num_stages: int,
        pp_group: dist.ProcessGroup,
        tp_mesh: DeviceMesh,
        static_stage_meta: dict[
            int, tuple[DTensor, DTensor, DTensor | None, DTensor | None]
        ]
        | None,
    ) -> PipelineStage:
        stage_module = pp_model.get_submodule(f"layers.{stage_index}")
        if static_stage_meta is None:
            return PipelineStage(
                submodule=stage_module,
                stage_index=stage_index,
                num_stages=num_stages,
                device=self.device,
                group=pp_group,
                get_mesh=lambda _dim_names, _layout: tp_mesh,
            )

        input_args, output_args, input_grads, output_grads = static_stage_meta[
            stage_index
        ]

        return PipelineStage(
            submodule=stage_module,
            stage_index=stage_index,
            num_stages=num_stages,
            device=self.device,
            group=pp_group,
            input_args=input_args,
            output_args=output_args,
            input_grads=input_grads,
            output_grads=output_grads,
            get_mesh=lambda _dim_names, _layout: tp_mesh,
        )

    def _run_pipeline_once(
        self,
        pp_model: NormMLPStack,
        input_dt: DTensor,
        target_dt: DTensor,
        pp_group: dist.ProcessGroup,
        pp_rank: int,
        num_stages: int,
        tp_mesh: DeviceMesh,
        static_stage_meta: dict[
            int, tuple[DTensor, DTensor, DTensor | None, DTensor | None]
        ]
        | None,
    ) -> None:
        pp_model.zero_grad(set_to_none=True)

        stage = self._make_stage_for_pp_run(
            pp_model,
            stage_index=pp_rank,
            num_stages=num_stages,
            pp_group=pp_group,
            tp_mesh=tp_mesh,
            static_stage_meta=static_stage_meta,
        )

        schedule = Schedule1F1B(
            stage,
            n_microbatches=n_microbatches,
            loss_fn=_loss_fn,
        )

        if stage.is_first:
            schedule.step(input_dt)
        elif stage.is_last:
            losses: list[torch.Tensor] = []
            _ = schedule.step(target=target_dt, losses=losses)
        else:
            schedule.step()

    def _assert_grad_close(
        self,
        pp_grad: torch.Tensor,
        ref_grad: torch.Tensor,
        param_fqn: str,
    ) -> None:
        pp_for_compare = pp_grad
        ref_for_compare = ref_grad

        if isinstance(pp_grad, DTensor) and isinstance(ref_grad, DTensor):
            pp_meta = _DTensorMeta.from_dtensor(pp_grad)
            ref_meta = _DTensorMeta.from_dtensor(ref_grad)
            self.assertEqual(
                pp_meta.get_diff(ref_meta),
                [],
                f"DTensor metadata mismatch for {param_fqn}",
            )
            pp_for_compare = pp_grad.to_local()
            ref_for_compare = ref_grad.to_local()

        torch.testing.assert_close(
            pp_for_compare,
            ref_for_compare,
            msg=f"Gradient mismatch for {param_fqn}",
        )

    def _assert_stage_grad_parity(
        self,
        pp_model: NormMLPStack,
        ref_grads: dict[str, torch.Tensor | None],
        stage_index: int,
    ) -> None:
        stage_module = pp_model.get_submodule(f"layers.{stage_index}")
        for name, param in stage_module.named_parameters():
            param_fqn = f"layers.{stage_index}.{name}"
            self.assertIsNotNone(param.grad, f"Missing PP grad for {param_fqn}")
            self.assertIsNotNone(
                ref_grads[param_fqn], f"Missing reference grad for {param_fqn}"
            )
            pp_grad = cast(torch.Tensor, param.grad)
            ref_grad = cast(torch.Tensor, ref_grads[param_fqn])
            self._assert_grad_close(pp_grad, ref_grad, param_fqn)

    def _run_numerical_correctness(
        self,
        apply_tp: Callable,
        placements: list,
        *,
        use_presharded_pp_input: bool = False,
        use_static_metadata: bool = True,
    ) -> None:
        self.init_pg()

        mesh = self._make_mesh()
        tp_mesh = mesh["tp"]
        pp_mesh = mesh["pp"]
        pp_group = pp_mesh.get_group()
        pp_rank = pp_mesh.get_local_rank()

        ref_model, pp_model = self._build_baseline_and_clones(
            n_layers=2,
            tp_mesh=tp_mesh,
            apply_tp=apply_tp,
        )

        input_full, target_full = self._make_full_batch_tensors()

        ref_input = self._wrap_full_tensor_as_dtensor(
            input_full.clone(),
            tp_mesh,
            placements,
        )
        ref_input.requires_grad_(True)
        ref_target = self._wrap_full_tensor_as_dtensor(
            target_full.clone(),
            tp_mesh,
            placements,
        )

        ref_grads, static_stage_meta = self._run_reference_microbatched(
            ref_model,
            ref_input,
            ref_target,
        )
        pp_static_stage_meta = static_stage_meta if use_static_metadata else None

        if use_presharded_pp_input:
            pp_input = self._make_presharded_input_dtensor(
                input_full.clone(),
                tp_mesh,
            )
        else:
            pp_input = self._wrap_full_tensor_as_dtensor(
                input_full.clone(),
                tp_mesh,
                placements,
            )
        pp_input.requires_grad_(True)
        pp_target = self._wrap_full_tensor_as_dtensor(
            target_full.clone(),
            tp_mesh,
            placements,
        )

        self._run_pipeline_once(
            pp_model=pp_model,
            input_dt=pp_input,
            target_dt=pp_target,
            pp_group=pp_group,
            pp_rank=pp_rank,
            num_stages=2,
            tp_mesh=tp_mesh,
            static_stage_meta=pp_static_stage_meta,
        )

        self._assert_stage_grad_parity(
            pp_model=pp_model,
            ref_grads=ref_grads,
            stage_index=pp_rank,
        )


class TestStaticMode(DTensorPPIntegrationBase):
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    def test_static_mode_replicate(self):
        self._run_numerical_correctness(
            apply_tp_replicate,
            [Replicate()],
            use_static_metadata=True,
        )

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    def test_static_mode_shard(self):
        self._run_numerical_correctness(
            apply_tp_shard,
            [Shard(1)],
            use_presharded_pp_input=True,
            use_static_metadata=True,
        )


class TestDynamicMode(DTensorPPIntegrationBase):
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    def test_dynamic_mode_replicate(self):
        self._run_numerical_correctness(
            apply_tp_replicate,
            [Replicate()],
            use_static_metadata=False,
        )

    @requires_accelerator_dist_backend(["nccl", "xccl"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR, f"{backend} test requires 4+ GPUs"
    )
    def test_dynamic_mode_shard(self):
        self._run_numerical_correctness(
            apply_tp_shard,
            [Shard(1)],
            use_presharded_pp_input=True,
            use_static_metadata=False,
        )


if __name__ == "__main__":
    run_tests()

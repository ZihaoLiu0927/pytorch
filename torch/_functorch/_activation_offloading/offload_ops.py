"""
Custom ops for async activation offloading between GPU and CPU.

These ops encapsulate stream management internally, producing a clean 2-node
IR pattern (offload/reload + wait) similar to c10d functional collectives.

Streams and events are managed by torchdynamo via new_stream()/new_event() and
passed as integer indices into the graph's external object table. This ensures
streams are created once and reused across iterations.

Instead of 7 nodes for offload:
    record_event -> fork -> wait_event -> record_stream -> device_put -> record_event -> join

We get 2 nodes:
    async_cpu = ao.offload(transfer_stream_idx, completion_event_idx, gpu_tensor)
    cpu_tensor = ao.wait(current_stream_idx, completion_event_idx, async_cpu)

Similarly for reload, instead of 5 nodes:
    fork -> wait_stream -> device_put -> record_event -> join

We get 2 nodes:
    async_gpu = ao.reload(transfer_stream_idx, completion_event_idx, cpu_tensor, device)
    gpu_tensor = ao.wait(current_stream_idx, completion_event_idx, async_gpu)
"""

import torch
from torch._dynamo.variables.streams import _get_event_by_index, _get_stream_by_index
from torch._library.custom_ops import custom_op
from torch.fx import has_side_effect


@custom_op("ao::offload", mutates_args=())
def offload(
    transfer_stream_idx: int, completion_event_idx: int, tensor: torch.Tensor
) -> torch.Tensor:
    """Async offload a GPU tensor to CPU on a dedicated transfer stream."""
    transfer_stream = _get_stream_by_index(transfer_stream_idx)
    completion_event = _get_event_by_index(completion_event_idx)
    current_stream = torch.accelerator.current_stream(tensor.device)

    transfer_stream.wait_stream(current_stream)
    tensor.record_stream(transfer_stream)

    torch.accelerator.set_stream(transfer_stream)
    result = tensor.to("cpu", non_blocking=True)
    torch.accelerator.set_stream(current_stream)

    transfer_stream.record_event(completion_event)

    return result


@offload.register_fake
def _(
    transfer_stream_idx: int, completion_event_idx: int, tensor: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(tensor, device="cpu")


@custom_op("ao::reload", mutates_args=())
def reload(
    transfer_stream_idx: int,
    completion_event_idx: int,
    tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Async reload a CPU tensor to GPU on a dedicated transfer stream."""
    transfer_stream = _get_stream_by_index(transfer_stream_idx)
    completion_event = _get_event_by_index(completion_event_idx)
    current_stream = torch.accelerator.current_stream(device)

    transfer_stream.wait_stream(current_stream)

    torch.accelerator.set_stream(transfer_stream)
    result = tensor.to(device, non_blocking=True)
    torch.accelerator.set_stream(current_stream)

    transfer_stream.record_event(completion_event)

    return result


@reload.register_fake
def _(
    transfer_stream_idx: int,
    completion_event_idx: int,
    tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty_like(tensor, device=device)


# ao::wait is defined via torch.library with an aliasing schema so the output
# can alias the input (custom_op forbids this).
_lib = torch.library.Library("ao", "DEF")
_lib.define("wait(int current_stream_idx, int completion_event_idx, Tensor(a) tensor) -> Tensor(a)")


@torch.library.impl("ao::wait", "cuda")
def _ao_wait_cuda(
    current_stream_idx: int, completion_event_idx: int, tensor: torch.Tensor
) -> torch.Tensor:
    completion_event = _get_event_by_index(completion_event_idx)
    current_stream = _get_stream_by_index(current_stream_idx)
    current_stream.wait_event(completion_event)
    return tensor


@torch.library.impl("ao::wait", "cpu")
def _ao_wait_cpu(
    current_stream_idx: int, completion_event_idx: int, tensor: torch.Tensor
) -> torch.Tensor:
    return tensor


@torch.library.register_fake("ao::wait")
def _ao_wait_fake(
    current_stream_idx: int, completion_event_idx: int, tensor: torch.Tensor
) -> torch.Tensor:
    return tensor


has_side_effect(torch.ops.ao.wait.default)


def wait(
    current_stream_idx: int, completion_event_idx: int, tensor: torch.Tensor
) -> torch.Tensor:
    """Callable wrapper so ``wait`` can be imported by name for op registration."""
    return torch.ops.ao.wait.default(current_stream_idx, completion_event_idx, tensor)

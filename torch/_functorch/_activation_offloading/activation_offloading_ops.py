"""Custom ops for async activation offloading between GPU and CPU.

These ops manage CUDA stream operations internally, running data transfers on a
dedicated transfer stream to overlap with compute on the default stream.

A single transfer stream per device is shared across all offload/reload ops.
Since offloads (forward) and reloads (backward) are serialized on this stream,
reloads automatically wait for prior offloads to complete.

Graph pattern per activation:
    Forward:  cpu = sixlib.ao_offload(gpu_tensor)
    Backward: gpu = sixlib.ao_reload(cpu, device)
              gpu = sixlib.ao_wait(gpu)       # aliases input, syncs stream
              ... use gpu ...
"""

import torch
from torch._library.custom_ops import custom_op
from torch.fx import has_side_effect

_transfer_streams: dict[str, torch.cuda.Stream] = {}


def _get_transfer_stream(device: torch.device) -> torch.cuda.Stream:
    """Get or create the dedicated transfer stream for the given device."""
    key = str(device)
    if key not in _transfer_streams:
        _transfer_streams[key] = torch.cuda.Stream(device)
    return _transfer_streams[key]


@custom_op("sixlib::ao_offload", mutates_args=())
def ao_offload(tensor: torch.Tensor) -> torch.Tensor:
    """Async offload a GPU tensor to CPU on a dedicated transfer stream.

    The transfer stream waits for the current (compute) stream to finish
    producing the tensor, then copies it to CPU asynchronously.
    ``record_stream`` prevents the GPU allocator from reusing the tensor's
    memory before the copy completes.
    """
    device = tensor.device
    stream = _get_transfer_stream(device)
    current = torch.cuda.current_stream(device)

    stream.wait_stream(current)
    tensor.record_stream(stream)

    torch.cuda.set_stream(stream)
    result = tensor.to("cpu", non_blocking=True)
    torch.cuda.set_stream(current)

    return result


@ao_offload.register_fake
def _(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(tensor, device="cpu")


@custom_op("sixlib::ao_reload", mutates_args=())
def ao_reload(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Async reload a CPU tensor to GPU on a dedicated transfer stream.

    The transfer stream waits for the current (compute) stream, then copies
    the tensor to the target device asynchronously.  Because offloads and
    reloads share the same transfer stream, the reload implicitly waits for
    any prior offload on that stream to finish.
    """
    stream = _get_transfer_stream(device)
    current = torch.cuda.current_stream(device)

    stream.wait_stream(current)

    torch.cuda.set_stream(stream)
    result = tensor.to(device, non_blocking=True)
    torch.cuda.set_stream(current)

    return result


@ao_reload.register_fake
def _(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.empty_like(tensor, device=device)


# ---------------------------------------------------------------------------
# ao_wait: defined via torch.library with an aliasing schema so the output
# can alias the input (custom_op forbids this).  Marked has_side_effect to
# prevent DCE — the stream synchronisation is a meaningful side effect.
# ---------------------------------------------------------------------------
_lib = torch.library.Library("sixlib", "DEF")
_lib.define("ao_wait(Tensor(a) tensor) -> Tensor(a)")


@torch.library.impl("sixlib::ao_wait", "cuda")
def _ao_wait_cuda(tensor: torch.Tensor) -> torch.Tensor:
    device = tensor.device
    stream = _get_transfer_stream(device)
    current = torch.cuda.current_stream(device)
    current.wait_stream(stream)
    return tensor


@torch.library.impl("sixlib::ao_wait", "cpu")
def _ao_wait_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


@torch.library.register_fake("sixlib::ao_wait")
def _ao_wait_fake(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


has_side_effect(torch.ops.sixlib.ao_wait.default)

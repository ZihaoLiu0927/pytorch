from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Concatenate, ParamSpec, TypeVar

import torch.library


P = ParamSpec("P")
R = TypeVar("R")

_OpOverrideFn = Callable[Concatenate[torch.DispatchKeySet, P], R]
_OpReplaceFn = Callable[P, R]

_OpFn = _OpOverrideFn | _OpReplaceFn


@dataclass
class _OverrideNode:
    """
    Track function override data
    """

    dsl_name: str
    override_fn: _OpFn
    unconditional_override: bool = False
    active: bool = True


# Store torch.library.Library instances
# libs: dict[[str, str, str], torch.library.Library] = {}
_libs: dict[tuple[str, str], torch.library.Library] = {}

# store graph structures
_GraphsType = dict[tuple[str, str], list[_OverrideNode]]
_graphs: _GraphsType = {}  # dict[[str, str], list[_OverrideNode]] = {}

_MappingType = dict[str, list[tuple[str, str]]]

# map a {dsl, op, dispatch_key} to keys to all graphs that contain it
_dsl_name_to_lib_graph: _MappingType = {}
_dispatch_key_to_lib_graph: _MappingType = {}
_op_symbol_to_lib_graph: _MappingType = {}


def _print_override_graphs(*, print_inactive: bool = False) -> None:
    for (op, key), node_list in _graphs.items():
        print(f"{op=}, {key=}")

        for i, node in enumerate(node_list):
            if node.active or print_inactive:
                s: str = f"    {i}: {node.dsl_name=}, {node.unconditional_override=}"
                if print_inactive:
                    s += f" {node.active=}"

                print(s)


def _get_library(op_symbol: str, dispatch_key: str) -> torch.library.Library:
    global _libs

    key = (op_symbol, dispatch_key)
    if key not in _libs:
        _libs[key] = torch.library.Library("aten", "IMPL", dispatch_key)

    return _libs[key]


def _resolve_iterable(iterable: str | Iterable[str] | None) -> Iterable[str]:
    if not iterable:
        return ()

    if not isinstance(iterable, Iterable) or isinstance(iterable, str):
        return (iterable,)

    return iterable


def _filter(
    dsl_name: str,
    op_symbol: str,
    dispatch_key: str,
    filter_dsl_names: str | Iterable[str] | None = None,
    filter_op_symbols: str | Iterable[str] | None = None,
    filter_dispatch_keys: str | Iterable[str] | None = None,
) -> bool:
    if (
        (not filter_dsl_names)
        and (not filter_op_symbols)
        and (not filter_dispatch_keys)
    ):
        raise ValueError("Must pass 1+ of filter_{dsl_names,op_symbols,dispatch_keys}")

    _filter_dsl_names = _resolve_iterable(filter_dsl_names)
    _filter_op_symbols = _resolve_iterable(filter_op_symbols)
    _filter_dispatch_keys = _resolve_iterable(filter_dispatch_keys)

    if dsl_name in _filter_dsl_names:
        return True

    if op_symbol in _filter_op_symbols:
        return True

    if dispatch_key in _filter_dispatch_keys:
        return True

    return False


def _build_key_set(
    disable_dsl_names: str | list[str] | None = None,
    disable_op_symbols: str | list[str] | None = None,
    disable_dispatch_keys: str | list[str] | None = None,
) -> set:
    key_set: set = set()

    def _append_to_set(
        disable: str | list[str] | None, graph_lib_dict: _MappingType
    ) -> None:
        disable_keys = _resolve_iterable(disable)

        for disable_key in disable_keys:
            for key in graph_lib_dict[disable_key]:
                key_set.add(key)

    _append_to_set(disable_dsl_names, _dsl_name_to_lib_graph)
    _append_to_set(disable_op_symbols, _op_symbol_to_lib_graph)
    _append_to_set(disable_dispatch_keys, _dispatch_key_to_lib_graph)

    return key_set


def _deregister_op_overrides(
    *,
    disable_dsl_names: str | list[str] | None = None,
    disable_op_symbols: str | list[str] | None = None,
    disable_dispatch_keys: str | list[str] | None = None,
) -> None:
    """
    De-register overrides from a given backend by deleting the
    associated `torch.library.Library` instance
    """
    global _libs

    # Need to resolve each of the `disable_*` arguments in an ideally
    # optimal way.
    # Libraries are stored in a dict[op_symbol, dispatch_key], but we also
    # have mappings from {dsl_name, op_symbol, dispatch_key} -> keys
    # Because key: list[str, str], we can just create a set of keys and iterate
    # over that.

    key_set: set = _build_key_set(
        disable_dsl_names, disable_op_symbols, disable_dispatch_keys
    )

    for key in key_set:
        op_symbol, dispatch_key = key
        # Remove the old graph
        del _libs[key]
        # create a new graph
        lib = _get_library(*key)

        # Re-register
        for node in _graphs[key]:
            filter_node = _filter(
                node.dsl_name,
                op_symbol,
                dispatch_key,
                disable_dsl_names,
                disable_op_symbols,
                disable_dispatch_keys,
            )
            if not filter_node:
                lib.impl(
                    "aten",
                    node.override_fn,
                    dispatch_key,
                    with_keyset=True,
                    allow_override=True,
                )
                node.active = True
            else:
                node.active = False


def _update_registration_maps(
    dsl_name: str,
    op_symbol: str,
    dispatch_key: str,
    key: tuple[str, str],
) -> None:
    global _dsl_name_to_lib_graph
    global _op_symbol_to_lib_graph
    global _dispatch_key_to_lib_graph

    def _get_new_entry_or_append(registration, symbol, key) -> None:
        l = registration.get(symbol, None)

        if l is None:
            l = [
                key,
            ]
        else:
            l.append(key)
        registration[symbol] = l

    _get_new_entry_or_append(_dsl_name_to_lib_graph, dsl_name, key)
    _get_new_entry_or_append(_op_symbol_to_lib_graph, op_symbol, key)
    _get_new_entry_or_append(_dispatch_key_to_lib_graph, dispatch_key, key)


def _register_op_override(
    backend: str,
    lib_symbol: str,
    op_symbol: str,
    dispatch_key: str,
    impl: _OpOverrideFn | _OpReplaceFn,
    *,
    allow_multiple_override=False,
    unconditional_override=False,
) -> None:
    """
    Register a passed override function to the dispatcher, based on the
    passed lib and op symbols, and the dispatch key.

    lib_symbol: str - library yourve overriding symbols in (generally "aten")
    op_symbol: str - name of the op you're overriding
    dispatch_key: str - dispatch key to override
    impl: Fn - implementation for the override
    allow_multiple_override: bool - allow overriding an existing override
    unconditional_override: bool - Impl doesn't have a fallback, and doesn't require
                                   torch.DispatchKeySet as the first argument.
    """
    # lib = _get_library(backend, lib_symbol, dispatch_key)
    key = (op_symbol, dispatch_key)
    lib = _get_library(*key)

    global _graphs
    op_graph = _graphs.get(key, [])

    op_graph.append(
        _OverrideNode(
            dsl_name=backend,
            override_fn=impl,
            unconditional_override=unconditional_override,
        )
    )
    _graphs[key] = op_graph
    # Build additional maps helpful for de-registration
    _update_registration_maps(backend, op_symbol, dispatch_key, key=key)

    lib.impl(
        op_symbol,
        impl,
        dispatch_key,
        with_keyset=(not unconditional_override),
        allow_override=allow_multiple_override,
    )

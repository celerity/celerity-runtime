import gdb
from typing import Iterable, Tuple


def iter_child_values(value: gdb.Value) -> Iterable[gdb.Value]:
    for _, child in gdb.default_visualizer(value).children():
        yield child


def deref_unique_ptr(unique_ptr: gdb.Value) -> gdb.Value:
    children = list(iter_child_values(unique_ptr))
    assert len(children) == 1
    return children[0].dereference()


def get_variant_content(variant: gdb.Value) -> gdb.Value:
    vis = gdb.default_visualizer(variant)
    # iterating over children() would dereference the contained value twice
    if hasattr(vis, 'contained_value'):
        return vis.contained_value
    else:
        return vis._contained_value


class StrongTypeAliasPrinter:
    def __init__(self, prefix: str, val: gdb.Value):
        self.prefix = prefix
        self.value = val['value']
    
    def to_string(self) -> str:
        return self.prefix + str(self.value)


class AllocationIdPrinter:
    def __init__(self, val: gdb.Value):
        self.mid = val['m_mid']
        self.raid = val['m_raid']
        self.is_null = self.mid == 0 and self.raid == 0

    def to_string(self) -> str:
        return 'M{}.A{}'.format(self.mid, self.raid) if not self.is_null else 'null'


class AllocationWithOffsetPrinter:
    def __init__(self, val: gdb.Value):
        self.id = val['id']
        self.offset_bytes = int(val['offset_bytes'])

    def to_string(self) -> str:
        if self.offset_bytes > 0:
            return '{} + {} bytes'.format(self.id, self.offset_bytes)
        else:
            return str(self.id)


class TransferIdPrinter:
    def __init__(self, val: gdb.Value):
        self.consumer_tid = val['consumer_tid']
        self.bid = val['bid']
        self.rid = val['rid']

    def to_string(self) -> str:
        s = '{}.{}'.format(self.consumer_tid, self.bid)
        if self.rid['value'] != 0:
            s += '.{}'.format(self.rid)
        return s


class CoordinatePrinter:
    def __init__(self, val: gdb.Value):
        self.values = val['m_values']['values']

    def to_string(self) -> str:
        i_min, i_max = self.values.type.range()
        return '[' + ', '.join(str(self.values[i]) for i in range(i_min, i_max+1)) + ']'


class SubrangePrinter:
    def __init__(self, val: gdb.Value):
        self.offset = val['offset']
        self.range = val['range']

    def to_string(self) -> str:
        return str(self.offset) + ' + ' + str(self.range)


class NdRangePrinter:
    def __init__(self, val):
        self.global_range = val['m_global_range']
        self.local_range = val['m_local_range']
        self.offset = val['m_offset']

    def children(self) -> Iterable[Tuple[str, gdb.Value]]:
        yield 'global_range', self.global_range
        yield 'local_range', self.local_range
        yield 'offset', self.offset


class ChunkPrinter:
    def __init__(self, val: gdb.Value):
        self.offset = val['offset']
        self.range = val['range']
        self.global_size = val['global_size']

    def children(self) -> Iterable[Tuple[str, gdb.Value]]:
        yield 'offset', self.offset
        yield 'range', self.range
        yield 'global_size', self.global_size


class BoxPrinter:
    def __init__(self, val: gdb.Value):
        self.min = val['m_min']
        self.max = val['m_max']

    def to_string(self) -> str:
        return str(self.min) + ' - ' + str(self.max)


class RegionPrinter:
    def __init__(self, val: gdb.Value):
        self.boxes = val['m_boxes']

    def to_string(self) -> str:
        if next(gdb.default_visualizer(self.boxes).children(), None) is None:
            return '{}'

    def children(self) -> Iterable[gdb.Value]:
        return gdb.default_visualizer(self.boxes).children()

    def display_hint(self) -> str:
        return 'array'


class RegionMapPrinter:
    def __init__(self, val: gdb.Value):
        impl = get_variant_content(val['m_region_map'])
        self.dims = int(val['m_dims'])
        if self.dims == 0:
            self.extent = '0d'
            self.value = impl['m_value']
        else:
            self.extent = impl['m_extent']
            self.root = deref_unique_ptr(impl['m_root'])

    def to_string(self) -> str:
        return 'region_map({})'.format(self.extent)

    def children(self) -> Iterable[Tuple[str, gdb.Value]]:
        if self.dims == 0:
            yield from [('value', self.value)]
            return

        def recurse_tree(root: gdb.Value):
            child_boxes = root['m_child_boxes']
            children = root['m_children']
            for box, child in zip(iter_child_values(child_boxes), iter_child_values(children)):
                child = get_variant_content(child)
                if 'celerity::detail::region_map_detail::inner_node' in str(child.type):
                    yield from recurse_tree(deref_unique_ptr(child))
                else:
                    yield ('[' + str(box) + ']', child)
        yield from recurse_tree(self.root)


class WriteCommandStatePrinter:
    def __init__(self, val: gdb.Value):
        bits = int(val['m_cid']['value'])
        self.cid = (bits & 0x3fff_ffff_ffff_ffff)
        self.fresh = (bits & 0x8000_0000_0000_0000) == 0
        self.replicated = (bits & 0x4000_0000_0000_0000) != 0

    def to_string(self) -> str:
        return 'C{} ({}{})'.format(self.cid,
                                   'fresh' if self.fresh else 'stale',
                                   ', replicated' if self.replicated else '')


def add_strong_type_alias_printer(pp: gdb.printing.RegexpCollectionPrettyPrinter, type: str, prefix: str):
    pp.add_printer(type, '^celerity::detail::{}$'.format(type), lambda val: StrongTypeAliasPrinter(prefix, val))


def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("Celerity")
    add_strong_type_alias_printer(pp, 'task_id', 'T')
    add_strong_type_alias_printer(pp, 'buffer_id', 'B')
    add_strong_type_alias_printer(pp, 'node_id', 'N')
    add_strong_type_alias_printer(pp, 'command_id', 'C')
    add_strong_type_alias_printer(pp, 'collective_group_id', 'CG')
    add_strong_type_alias_printer(pp, 'reduction_id', 'R')
    add_strong_type_alias_printer(pp, 'host_object_id', 'H')
    add_strong_type_alias_printer(pp, 'hydration_id', 'HY')
    add_strong_type_alias_printer(pp, 'memory_id', 'M')
    add_strong_type_alias_printer(pp, 'device_id', 'D')
    add_strong_type_alias_printer(pp, 'raw_allocation_id', 'A')
    add_strong_type_alias_printer(pp, 'instruction_id', 'I')
    add_strong_type_alias_printer(pp, 'message_id', 'MSG')
    pp.add_printer('allocation_id', '^celerity::detail::allocation_id$', AllocationIdPrinter)
    pp.add_printer('allocation_with_offset', '^celerity::detail::allocation_with_offset$', AllocationWithOffsetPrinter)
    pp.add_printer('id', '^celerity::id<.*>$', CoordinatePrinter)
    pp.add_printer('range', '^celerity::range<.*>$', CoordinatePrinter)
    pp.add_printer('subrange', '^celerity::subrange<.*>$', SubrangePrinter)
    pp.add_printer('nd_range', '^celerity::nd_range<.*>$', NdRangePrinter)
    pp.add_printer('chunk', '^celerity::chunk<.*>$', ChunkPrinter)
    pp.add_printer('box', '^celerity::detail::box<.*>$', BoxPrinter)
    pp.add_printer('region', '^celerity::detail::region<.*>$', RegionPrinter)
    pp.add_printer('region_map', '^celerity::detail::region_map<.*>$', RegionMapPrinter)
    pp.add_printer('write_command_state', '^celerity::detail::write_command_state$', WriteCommandStatePrinter)
    pp.add_printer('transfer_id', '^celerity::detail::transfer_id$', TransferIdPrinter)

    return pp


pretty_printer = build_pretty_printer()

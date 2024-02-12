#!/usr/bin/env python3

import subprocess
import sys

EXPECTED = b'''__builtin_trap();
tid = T10
bid = B11
nid = N12
cid = C13
cgid = CG14
rid = R15
hoid = H16
hyid = HY17
trid = T18.B19
reduction_trid = T20.B21.R22
mid = M23
raid = A24
did = D25
iid = I26
aid = M27.A28
alloc = M29.A30
alloc_offset = M31.A32 + 33 bytes
msgid = MSG34
id = [1, 2, 3]
range = [1, 2, 3]
subrange = [1, 2, 3] + [4, 5, 6]
chunk = {offset = [1, 2, 3], range = [4, 5, 6], global_size = [7, 8, 9]}
nd_range = {global_range = [2, 4, 6], local_range = [1, 2, 3], offset = [7, 8, 9]}
box = [1, 2, 3] - [4, 5, 6]
empty_region = {}
region = {[1, 2, 3] - [4, 5, 6], [11, 2, 3] - [14, 5, 6], [21, 2, 3] - [24, 5, 6]}
region_map = region_map([0, 0, 0] - [10, 10, 10]) = {[[0, 0, 0] - [10, 10, 1]] = 0, [[0, 0, 5] - [10, 10, 10]] = 0, [[0, 0, 1] - [10, 1, 5]] = 0, [[0, 5, 1] - [10, 10, 5]] = 0, [[0, 1, 1] - [1, 5, 5]] = 0, [[1, 1, 1] - [2, 2, 2]] = 1337, [[1, 1, 2] - [3, 3, 3]] = 69, [[1, 2, 1] - [3, 3, 2]] = 69, [[2, 1, 1] - [3, 2, 2]] = 69, [[1, 1, 3] - [5, 5, 5]] = 42, [[3, 1, 1] - [5, 3, 3]] = 42, [[1, 3, 1] - [5, 5, 3]] = 42, [[5, 1, 1] - [10, 5, 5]] = 0}
region_map_0d = region_map(0d) = {value = 42}
wcs_fresh = C123 (fresh)
wcs_stale = C123 (stale)
wcs_replicated = C123 (fresh, replicated)
'''

# invoke as `pretty-print-test.py build/test/debug/pretty_printables`
assert len(sys.argv) == 2

out = subprocess.check_output(['gdb', '-batch', sys.argv[1], '-ex', 'r', '-ex', 'i locals'])
if not out.endswith(EXPECTED):
    print('-*- pretty-print test FAILED: expected GDB output to end with', file=sys.stderr)
    print(EXPECTED.decode('utf-8'), file=sys.stderr)
    print('-*- but got', file=sys.stderr)
    print(out.decode('utf-8'), file=sys.stderr)
    print('-*-', file=sys.stderr)
    sys.exit(1)

#!/usr/bin/env python3

import sys
import struct
from enum import Enum

EPSILON = 1e-12

def read_tile(f):
    header = f.read(12)
    if len(header) == 0:
        return None
    x, y, n = struct.unpack("3I", header)
    points = []
    for i in range(n):
        points.append(struct.unpack("3d", f.read(24)))
    return (x, y, points)

class Result(Enum):
    MISMATCH = 1
    EXACT = 2
    APPROXIMATE = 3

def compare_tiles(t1, t2) -> Result:
    if t1[0] != t2[0] or t1[1] != t2[1]:
        print("Coordinates differ: {} vs {}".format(t1[:2], t2[:2]))
        return Result.MISMATCH
    if len(t1[2]) != len(t2[2]):
        print("Lengths differ: {} vs {}".format(len(t1[2]), len(t2[2])))
        return Result.MISMATCH
    p1 = sorted(t1[2])
    p2 = sorted(t2[2])
    result = Result.EXACT
    for i in range(len(p1)):
        if p1[i] != p2[i]:
            if abs(p1[i][0] - p2[i][0]) > EPSILON or abs(p1[i][1] - p2[i][1]) > EPSILON or abs(p1[i][2] - p2[i][2]) > EPSILON:
                print("Points differ: {} vs {}".format(p1[i], p2[i]))
                return Result.MISMATCH
            else:
                result = Result.APPROXIMATE
    return result

def main():
    if len(sys.argv) != 3:
        print("Usage: {} <file1> <file2>".format(sys.argv[0]))
        sys.exit(1)
    f1 = open(sys.argv[1], "rb")
    f2 = open(sys.argv[2], "rb")
    result = Result.EXACT
    while True:
        t1 = read_tile(f1)
        t2 = read_tile(f2)
        if t1 is None and t2 is None:
            break
        if t1 is None or t2 is None:
            print("Files are different: Different number of tiles")
            sys.exit(1)
        r = compare_tiles(t1, t2)
        if r == Result.MISMATCH:
            print("Files are different")
            sys.exit(1)
        if r == Result.APPROXIMATE:
            result = Result.APPROXIMATE
    if result == Result.EXACT:
        print("Files match exactly")
    else:
        print("Files match approximately (epsilon={})".format(EPSILON))

if __name__ == "__main__":
    main()

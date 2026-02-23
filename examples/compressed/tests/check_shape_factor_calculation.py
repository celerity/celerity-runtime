import argparse
import file_io

def read_tiles_shapefactor(filename_tiles, filename_shape):
    tiles = file_io.PointCloudGrid.from_file(filename_tiles).convert_to_point_cloud().points
    shape_factor = file_io.PointCloudGrid.from_file(filename_shape).convert_to_point_cloud().points

    return tiles, shape_factor

def map_tiles_to_shapefactor(tiles, shape_factor):
    tiles_shapefactor = {}

    for tile, shape in zip(tiles, shape_factor):
        tiles_shapefactor[tile] = shape

    return tiles_shapefactor

def check_shape_factor_lenght(tiles, shape_factor):
    if len(tiles) != len(shape_factor):
        print("Number of tiles and shapefactor are not the same")
        return False

    return True

def check_shape_factor_calculation(map_tiles_shapefactor, map_tiles_shapefactor_new, epsilon=1e-6):
    if not isinstance(map_tiles_shapefactor, dict) or not isinstance(map_tiles_shapefactor_new, dict):
        raise TypeError("Expected dictionaries for map_tiles_shapefactor and map_tiles_shapefactor_new")

    for tile, shape_factor in map_tiles_shapefactor.items():
        if tile not in map_tiles_shapefactor_new:
            print(f"Tile {tile} is not found in the new shapefactor")
            return False

        if not all(abs(shape_factor[i] - map_tiles_shapefactor_new[tile][i]) < epsilon for i in range(3)):
            print(f"Shapefactor for tile {tile} is not calculated correctly")
            print(f"The shapefactor for this tile should be {shape_factor} and is {map_tiles_shapefactor_new[tile]}")
            return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check if the shapefactor is calculated correctly.')
    parser.add_argument('filename_tiles_ground_truth', type=str, help='Filename of the tiles ground truth')
    parser.add_argument('filename_shape_ground_truth', type=str, help='Filename of the shapefactor ground truth')
    parser.add_argument('filename_tiles', type=str, help='Filename of the tiles to test')
    parser.add_argument('filename_shape', type=str, help='Filename of the shapefactor to test')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='Epsilon for comparing floating point numbers')

    args = parser.parse_args()

    RED = '\033[91m'
    GREEN = '\033[92m'
    END = '\033[0m'

    tiles, shape_factor = read_tiles_shapefactor(args.filename_tiles_ground_truth, args.filename_shape_ground_truth)
    map_tiles_shapefactor = map_tiles_to_shapefactor(tiles, shape_factor)

    tiles_test, shape_factor_test = read_tiles_shapefactor(args.filename_tiles, args.filename_shape)
    map_tiles_shapefactor_new = map_tiles_to_shapefactor(tiles_test, shape_factor_test)

    if check_shape_factor_lenght(tiles_test, shape_factor_test):
        print(f"{GREEN}Shapefactor lenght correct{END}")
    else:
        print(f"{RED}Shapefactor lenght not correct{END}")
        exit(1)

    if check_shape_factor_calculation(map_tiles_shapefactor, map_tiles_shapefactor_new, args.epsilon):
        print(f"{GREEN}All shapefactors are calculated correctly.{END}")
        exit(0)
    else:
        print(f"{RED}There are errors in the shapefactor calculations.{END}")
        exit(1)
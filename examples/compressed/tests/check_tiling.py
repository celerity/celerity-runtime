import argparse
import file_io

def check_missing_combinations(filename, range_x=(-100, 100), range_y=(0, 0), range_z=(0, 0)):
    coordinates = set(file_io.PointCloudGrid.from_file(filename).convert_to_point_cloud().points)

    all_combinations = set((x, y, z) for x in range(range_x[0], range_x[1] + 1) for y in range(range_y[0], range_y[1] + 1) for z in range(range_z[0], range_z[1] + 1))

    missing_combinations = all_combinations - coordinates

    return sorted(missing_combinations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check missing combinations in a file.')
    parser.add_argument('filename', type=str, help='The filename to check')
    parser.add_argument('--x_range', nargs=2, type=int, default=(-100, 100), help='Range for x values')
    parser.add_argument('--y_range', nargs=2, type=int, default=(0, 0), help='Range for y values')
    parser.add_argument('--z_range', nargs=2, type=int, default=(0, 0), help='Range for z values')

    args = parser.parse_args()

    missing_combinations = check_missing_combinations(filename=args.filename, range_x=tuple(args.x_range), range_y=tuple(args.y_range), range_z=tuple(args.z_range))

    RED = '\033[91m'
    GREEN = '\033[92m'
    ENDC = '\033[0m'

    if len(missing_combinations) > 0:
        print(f"\n{RED}Missing Combinations:{ENDC}")
        print(f"{RED}{'X':>5} {'Y':>5} {'Z':>5}{ENDC}") 
        for combo in missing_combinations[:10]:
            x, y, z = combo
            print(f"{RED}{x:>5} {y:>5} {z:>5}{ENDC}") 
        if len(missing_combinations) > 10:
            print(f"{RED}... and more{ENDC}")
        print(f"\n{RED}Total Number of Missing Combinations: {len(missing_combinations)}{ENDC}")
        exit(1)
    else:
        print(f"{GREEN}No missing combinations found.{ENDC}")
        exit(0)
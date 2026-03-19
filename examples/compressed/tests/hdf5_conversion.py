import h5py
import hdf5plugin
import numpy as np

def get_data_and_numerical_shifts_from_file(input_file, group_string, data_set_strings):
    with h5py.File(input_file, 'r') as f:
        group = f[group_string]
        data = []
        numerical_shifts = []
        # global_range = group.attrs['Range']
        global_range = 0

        for data_set_string in data_set_strings:
            dataset = group[data_set_string]
            data.append(dataset[()])
            numerical_shifts.append(dataset.attrs['Fiber::NumericalShift'][0])

        return (global_range, (data, numerical_shifts))

    return None

def shift_to_global_coordinates(numerical_shifts, data):
    global_data = []
    for i in range(len(numerical_shifts)):
        global_data.append(add_numerical_shift(convert_data(data[i]), numerical_shifts[i]))

    return global_data

def get_data(group, data_set_string):
    dataset = group[data_set_string]
    data = dataset[()]
    return (dataset, data)

def convert_data(data):
    return data.astype([('x', '<f8'), ('y', '<f8'), ('z', '<f8')])

def add_numerical_shift(data, numerical_shift):
    data['x'] = data['x'] + numerical_shift[0]
    data['y'] = data['y'] + numerical_shift[1]
    data['z'] = data['z'] + numerical_shift[2]
    return data

# center data around origin
def center_data(data):
    data['x'] = data['x'] - np.mean(data['x'])
    data['y'] = data['y'] - np.mean(data['y'])
    data['z'] = data['z'] - np.mean(data['z'])
    return data

def read_points(filename, search_string, data_sets = 8): 
    group_string = "/t=000000000.0000000000/"+ search_string + "/Points/StandardCartesianChart3D/Positions"

    # specific to this file -> should be passed as an argument 
    # instead of hardcoding it here make for loop to get all the data

    data = []

    for i in range(data_sets):
        data_set_string = "LasFragment_" + str(i)
        data.append(data_set_string)

    (_, (data, numerical_shifts)) = get_data_and_numerical_shifts_from_file(filename, group_string, data)

    data_global = shift_to_global_coordinates(numerical_shifts, data)
    points = np.concatenate(data_global)
    return points

    # return center_data(points)

def read_grid(filename):
    group_string = "/t=000000000.0000000000/MergedGrids/Points/StandardCartesianChart3D/Positions/"

    # specific to this file -> should be passed as an argument
    data_set_string_one = "RegularFragment[0x0x0]"
    data_set_string_two = "RegularFragment[0x1x0]"
    data_set_string_three = "RegularFragment[1x0x0]"
    data_set_string_four = "RegularFragment[1x1x0]"

    (_, (data, numerical_shifts)) = get_data_and_numerical_shifts_from_file(filename, group_string, [data_set_string_one, data_set_string_two, data_set_string_three, data_set_string_four])
    data_global = shift_to_global_coordinates(numerical_shifts, data)

    # tmp = np.concatenate(data_global)
    # mean = np.mean(tmp['x']), np.mean(tmp['y']), np.mean(tmp['z'])

    # for i in range(len(data_global)):
    #     data_global[i]['x'] = data_global[i]['x'] - mean[0]
    #     data_global[i]['y'] = data_global[i]['y'] - mean[1]
    #     data_global[i]['z'] = data_global[i]['z'] - mean[2]

    data_2D = []
    for i in range(2):
        data_2D.append([])
        for j in range(2):
            data_2D[i].append([])

    for i in range(len(data_global)):
        data_2D[i // 2][i % 2] = data_global[i]

    return data_2D

# sample usage
if __name__ == '__main__':
    input_file = '../../hdf5_umuguc/Merge_orig.f5'

    merged_points = read_points(input_file, "180530_075631_Scanner_1_0", 28)

    print(merged_points)
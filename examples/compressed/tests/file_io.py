import struct
import os

import hdf5_conversion

class PointCloudGrid:
    def __init__(self, points):
        self.points = points

    @staticmethod
    def read_txt_file(filename):
        grid = []
        with open(filename, 'r') as f:
            grid_index = None
            for line in f:
                line = line.strip()
                if line.startswith('('):
                    grid_index = tuple(map(int, line.strip('()\n').split(',')))

                    while len(grid) <= grid_index[0]:
                        grid.append([])

                    while len(grid[grid_index[0]]) <= grid_index[1]:
                        grid[grid_index[0]].append([])

                elif line:
                    point = tuple(map(float, line.split()))
                    grid[grid_index[0]][grid_index[1]].append(point)

        return grid
    
    @staticmethod
    def read_binary_file(filename, data_type='ddd'):
        grid = []
        with open(filename, 'rb') as binary_file:
            while True:
                grid_index_bytes = binary_file.read(struct.calcsize('ii'))
                if not grid_index_bytes:
                    break

                grid_index = struct.unpack('ii', grid_index_bytes)
                amount_bytes = binary_file.read(struct.calcsize('i'))
                amount = struct.unpack('i', amount_bytes)[0]

                while len(grid) <= grid_index[0]:
                    grid.append([])

                while len(grid[grid_index[0]]) <= grid_index[1]:
                    grid[grid_index[0]].append([])

                for _ in range(amount):
                    point_bytes = binary_file.read(struct.calcsize(data_type))
                    point = struct.unpack(data_type, point_bytes)
                    grid[grid_index[0]][grid_index[1]].append(point)

        return grid
    
    @staticmethod
    def read_hdf5_file(filename):
        points = hdf5_conversion.read_grid(filename, "180530_075631_Scanner_1_0", 28)
        return points
    
    @staticmethod
    def to_txt_file(filename, points):
        with open(filename, 'w') as f:
            for i in range(len(points)):
                for j in range(len(points[i])):
                    f.write(f"({i},{j})\n")
                    f.write("\n")
                    for point in points[i][j]:
                        f.write(" ".join(map(str, point)) + "\n")
                    f.write("\n")

    @staticmethod
    def to_binary_file(filename, points, data_type='ddd'):
        with open(filename, 'wb') as binary_file:
            for i in range(len(points)):
                for j in range(len(points[i])):
                    grid_index = struct.pack('ii', i, j)
                    binary_file.write(grid_index)
                    amount = len(points[i][j])
                    amount_bytes = struct.pack('i', amount)
                    binary_file.write(amount_bytes)

                    for point in points[i][j]:
                        point_bytes = struct.pack(data_type, *point)
                        binary_file.write(point_bytes)
    
    @staticmethod
    def default_function(filename):
        print("Extension not supported:", os.path.splitext(filename)[1])
        exit(1)

    @classmethod
    def from_file(cls, filename, datatype='ddd'):
        my_dict = {'.txt': cls.read_txt_file, '.bin': cls.read_binary_file, '.f5': cls.read_hdf5_file}
        points = my_dict.get(os.path.splitext(filename)[1], cls.default_function)(filename, datatype)
        return cls(points)
    
    def to_file(self, filename, datatype='ddd'):
        my_dict = {'.txt': self.to_txt_file, '.bin': self.to_binary_file}
        my_dict.get(os.path.splitext(filename)[1], self.default_function)(filename, self.points, datatype)

    def convert_to_point_cloud(self):
        points = []
        for i in range(len(self.points)):
            for j in range(len(self.points[i])):
                points.extend(self.points[i][j])
        return PointCloud(points)
    
    def convert_to_2D_grid(self):
        grid = []
        for i in range(len(self.points)):
            for j in range(len(self.points[i])):
                if self.points[i][j]:
                    grid.append(self.points[i][j])
        return grid

class PointCloud:
    def __init__(self, points):
        self.points = points

    @staticmethod
    def read_txt_file(filename, data_type='ddd'):
        points = []
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():
                    point = tuple(map(float, line.split()))
                    points.append(point)
        return points
    
    @staticmethod
    def read_binary_file(filename, data_type='ddd'):
        points = []
        with open(filename, 'rb') as binary_file:
            while True:
                point_bytes = binary_file.read(struct.calcsize(data_type))
                if not point_bytes:
                    break

                point = struct.unpack(data_type, point_bytes)
                points.append(point)

        return points
    
    @staticmethod
    def read_hdf5_file(filename):
        points = hdf5_conversion.read_points(filename, "180530_075631_Scanner_1_0", 28)
        return points
    
    @staticmethod
    def to_txt_file(filename, points, data_type='ddd'):
        with open(filename, 'w') as f:
            for point in points:
                f.write(" ".join(map(str, point)) + "\n")

    @staticmethod
    def to_binary_file(filename, points, data_type='ddd'):
        with open(filename, 'wb') as binary_file:
            for point in points:
                point_bytes = struct.pack(data_type, *point)
                binary_file.write(point_bytes)

    @staticmethod
    def default_function(filename):
        print("Extension not supported:", os.path.splitext(filename)[1])
        exit(1)

    @classmethod
    def from_file(cls, filename, datatype='ddd'):
        my_dict = {'.txt': cls.read_txt_file, '.bin': cls.read_binary_file, '.f5': cls.read_hdf5_file}
        points = my_dict.get(os.path.splitext(filename)[1], cls.default_function)(filename, datatype)
        return cls(points)
    
    def to_file(self, filename, datatype='ddd'):
        my_dict = {'.txt': self.to_txt_file, '.bin': self.to_binary_file}
        my_dict.get(os.path.splitext(filename)[1], self.default_function)(filename, self.points, datatype)

# Example usage
# if __name__ == "__main__":
#     point_cloud = PointCloudGrid.from_file('points_out.txt')
#     point_cloud.to_file('points_out_new.bin')
#     point_cloud_new = PointCloudGrid.from_file('points_out_new.bin')
#     print(point_cloud_new.points == point_cloud.points)

#     point_cloud = PointCloud.from_file('plane_2D_200_200.txt')
#     point_cloud.to_file('test.bin')
#     point_cloud_new = PointCloud.from_file('test.bin')
#     print(point_cloud_new.points == point_cloud.points)
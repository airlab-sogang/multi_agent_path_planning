import numpy as np

if __name__ == '__main__':
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
    # select data[i][2] > 0 and get max value of data[i][0] and data[i][1]

    # method 1
    mask = data[:, 2] > 0
    print(data[mask][:, :2].max(axis=0))

    # select data[i][2] > 0 and get min value of data[i][0] and data[i][1]
    print(data[mask][:, :2].min(axis=0))


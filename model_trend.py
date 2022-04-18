import sklearn
import argparse
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np

def plot_data(text_file):
    x_conf = []
    y_area = []

    with open(text_file, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line_split = line.split(',')
            x_conf.append([float(line_split[-3])])
            y_area.append([float(line_split[-2])])

    assert len(x_conf) == len(y_area)

    scaler = preprocessing.MinMaxScaler().fit(np.array(x_conf))
    x_conf = scaler.transform(np.array(x_conf))

    scaler = preprocessing.MinMaxScaler().fit(np.array(y_area))
    y_area = scaler.transform(np.array(y_area))




    fig = plt.figure(1)
    plt.title("area vs confidence", fontsize='16')
    plt.scatter(x_conf, y_area)
    plt.xlabel("Confidence",fontsize='13')
    plt.ylabel("Area",fontsize='13')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "text file")
    parser.add_argument('--text_fp', dest = 'text_fp', required = True)

    args = parser.parse_args()

    plot_data(args.text_fp)

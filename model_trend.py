# 1 / (Sqaure Root Area) on x-axis
# Graph tanhx


import sklearn
import argparse
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from scipy.optimize import curve_fit
import math

def test_func(x, a, b):
    return a * np.tanh(b * x)

def plot_data(text_files):
    text_files = text_files.split(',')
    x_conf = []
    y_area = []

    total_ntp_indices = list()

    for text_file in text_files:
        freq = {}
        with open(text_file, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                line_split = line.split(',')
                if line_split[0] not in freq:
                    freq[line_split[0]] = 1
                else:
                    freq[line_split[0]] += 1
                if ((line_split[1] == "total_ctp" or line_split[1] == "total_ntp") and freq[line_split[0]] < 3 and float(line_split[-2]) < 20000):
                    x_conf.append([float(line_split[-3])])
                    y_area.append([float(line_split[-2])])
                    if line_split[1] == "total_ntp":
                        total_ntp_indices.append(len(x_conf) - 1)

    # Plot d0 vs d3

    x_conf_d0, x_conf_d3 = [], []
    y_area_d0, y_area_d3 = [], []
    #
    # freq = {}
    # with open(text_files[0], "r") as f:
    #     for i, line in enumerate(f):
    #         if i == 0:
    #             continue
    #         line_split = line.split(',')
    #         if line_split[0] not in freq:
    #             freq[line_split[0]] = 1
    #         else:
    #             freq[line_split[0]] += 1
    #         if ((line_split[1] == "total_ntp" or line_split[1] == "total_ctp") and freq[line_split[0]] < 2 and float(line_split[-2]) < 2000):
    #             x_conf_d3.append([float(line_split[-3])])
    #             y_area_d3.append([float(line_split[-2])])
    #             if line_split[1] == "total_ctp":
    #                 total_ntp_indices.append(len(x_conf_d3) - 1)
    #
    # freq = {}
    # with open(text_files[1], "r") as f:
    #     for i, line in enumerate(f):
    #         if i == 0:
    #             continue
    #         line_split = line.split(',')
    #         if line_split[0] not in freq:
    #             freq[line_split[0]] = 1
    #         else:
    #             freq[line_split[0]] += 1
    #         if ((line_split[1] == "total_ntp" or line_split[1] == "total_ctp") and freq[line_split[0]] < 2 and float(line_split[-2]) < 2000):
    #             x_conf_d0.append([float(line_split[-3])])
    #             y_area_d0.append([float(line_split[-2])])
    #             if line_split[1] == "total_ctp":
    #                 total_ntp_indices.append(len(x_conf_d3) - 1)


    assert len(x_conf_d0) == len(y_area_d0)
    assert len(x_conf_d3) == len(y_area_d3)



    x_conf.extend(x_conf_d0)
    x_conf.extend(x_conf_d3)
    y_area.extend(y_area_d0)
    y_area.extend(y_area_d3)
    #
    # params, params_covariance = curve_fit(test_func, float(np.array(y_area)), float(np.array(x_conf)),
    #                                            p0=[2, 2])
    # print(params)


    fig = plt.figure(1)
    # Red is total_ntp
    # plt.scatter(y_area_d0, x_conf_d0, label = "EfficientDet d0", c = "blue", marker = 'x', s = 4)
    # plt.scatter(y_area_d3, x_conf_d3, label = "EfficientDet d3", c = "red", marker = 'x', s = 4)

    plt.scatter([v for i, v in enumerate(y_area) if i not in total_ntp_indices], [v for i, v in enumerate(x_conf) if i not in total_ntp_indices], c = "red", marker = "x", label = "ctp", s = 4)
    # Blue is total_ctp
    plt.scatter([v for i, v in enumerate(y_area) if i in total_ntp_indices], [v for i, v in enumerate(x_conf) if i in total_ntp_indices], c = "blue", marker = "x", label = "ntp", s = 4)
    plt.legend(loc="lower right")
    # # plt.plot(x_conf, y_avg)
    plt.xlabel("Object Area in Pixels",fontsize='13')
    plt.ylabel("Prediction Confidence",fontsize='13')
    # plt.savefig('ConfvArea.pdf')
    plt.grid()
    plt.show()


    # plt.savefig("AreaVConf.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "text file")
    parser.add_argument('--text_fp', dest = 'text_fp', required = True)

    args = parser.parse_args()

    plot_data(args.text_fp)

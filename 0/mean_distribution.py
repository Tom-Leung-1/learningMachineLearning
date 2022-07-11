import numpy as np
import math
import matplotlib.pyplot as plt

def estimate_mean(n, delta):
    # code here
    max, min, samples_no, samples_list = read_file("./data.txt")
    group_no = samples_no/n
    n_split = np.array_split(samples_list, group_no)
    size = n
    mean_list = sampling_mean(n_split, size)
    bin_no = math.ceil((max-min)/delta)
    bin_dict = {min + delta * x:get_max_range(min, max, delta, x) for x in range(bin_no)}
    bin_pmf_dict = {k:0 for k in bin_dict}
    for mean in mean_list:
        for min_r, max_r in bin_dict.items():
            if max_r == max:
                if mean >= min_r and mean <= max_r:
                    bin_pmf_dict[min_r] += 1
            else:
                if mean >= min_r and mean < max_r:
                    bin_pmf_dict[min_r] += 1
    groups_no = len(mean_list)
    bin_pmf_dict = {k:v/groups_no for k, v in bin_pmf_dict.items()}
    mean_pmf = 0
    for min_r, prob in bin_pmf_dict.items():
        mean_pmf += min_r * prob
    return bin_pmf_dict, mean_pmf


def make_figure(bin_pmf_dict, delta, i):
    x_data = [k for k in bin_pmf_dict]
    y_data = [bin_pmf_dict[k] for k in bin_pmf_dict]
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.bar(x_data, y_data, delta, edgecolor= 'black', align="edge")
    ax1.set_xlabel("Sample mean")
    ax1.set_ylabel("Probability")
    plt.savefig(str(i)+".png")


def get_max_range(min, max, delta, bin_place):
    # lower_range = min + delta * (bin_place)
    max_range = min + delta * (bin_place+1) if max > min + delta * (bin_place) else max
    return max_range

def sampling_mean(n_split, size):
    mean_list = []
    for array in n_split:
        total = np.sum(array)
        sampling_mean = total/size
        mean_list.append(sampling_mean)
    return mean_list


def read_file(path):
    f = open(path, "r")
    place = 1
    samples_no = 0
    max = 0
    min = 0
    samples_list = []
    for line in f:
        line = line.strip()
        if place > 3:
            samples_list.append(float(line))
        elif place == 1:
            samples_no = int(float(line))
            place += 1
        elif place == 2:
            min = float(line)
            place += 1
        elif place == 3:
            max = float(line)
            place += 1
    return max, min, samples_no, samples_list

"""
if __name__ == "__main__":
    # read_file("./data.txt")
    bin_pmf_dict_1, mean_pmf_1 = estimate_mean(20, 0.1)
    bin_pmf_dict_2, mean_pmf_2 = estimate_mean(100, 0.1)
    bin_pmf_dict_3, mean_pmf_3 = estimate_mean(20, 0.001)
    bin_pmf_dict_4, mean_pmf_4 = estimate_mean(100, 0.001)
    '''
    make_figure(bin_pmf_dict_1, 0.1, 1)
    make_figure(bin_pmf_dict_2, 0.1, 2)
    make_figure(bin_pmf_dict_3, 0.001, 3)
    make_figure(bin_pmf_dict_4, 0.001, 4)
    '''
    print("n =",20,", delta =", 0.1, ", mean_pmf is", mean_pmf_1)
    print("n =", 100, ", delta =", 0.1, ", mean_pmf is", mean_pmf_2)
    print("n =", 20, ", delta =", 0.001, ", mean_pmf is", mean_pmf_3)
    print("n =", 100, ", delta =", 0.001, ", mean_pmf is", mean_pmf_4)
"""
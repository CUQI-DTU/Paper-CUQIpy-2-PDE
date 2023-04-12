
# %%
import numpy as np
import matplotlib.pyplot as plt

def collect_all_keys(case_list):
    keys = []
    for case in case_list:
        keys.extend(case.keys())
    return set(keys)

def get_keys_for_values_that_are_not_the_same(case_list, keys):
    unique_keys = []
    for key in keys:
        values = []
        for case in case_list:
            try:
                values.append(case[key])
            except KeyError:
                pass
        include= False
        for value in values:
            if np.any(value != values[0]):
                include = True

        if include:
            unique_keys.append(key)
    return unique_keys

def print_parameters_that_are_not_the_same(case_list, ignore_keys=[], include_keys=[]):
    keys = collect_all_keys(case_list)
    unique_keys = get_keys_for_values_that_are_not_the_same(case_list, keys)
    for key in ignore_keys: unique_keys.remove(key)
    for key in include_keys: unique_keys.append(key)
    for key in unique_keys:
        print()
        print("###",key,"###")
        for case in case_list:
            try:
                print(case[key], "for case", case['case_name'])
            except KeyError:
                print("No key",key,"in case",case['case_name'])


def matplotlib_setup(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE):
    plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title 

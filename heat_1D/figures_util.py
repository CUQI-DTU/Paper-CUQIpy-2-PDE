
# %%
import numpy as np

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

    
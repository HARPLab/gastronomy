import numpy as np

import csv
import h5py
import pdb

from collections import OrderedDict

class ControlLoopData(object):
    def __init__(self, data_as_ord_dict):
        self.data_as_ord_dict = data_as_ord_dict
        self.skill_time_as_tuple_list = self.get_times_for_separate_skills(
                data_as_ord_dict)        

    def get_times_for_separate_skills(self, data_as_ord_dict):
        time_arr = data_as_ord_dict['time']
        start_times, _ = np.where(time_arr <= 0.0)
        end_times = [start_times[i] for i in range(1, len(start_times))]
        end_times.append(time_arr.shape[0])
        return list(zip(start_times, end_times))

    def get_duration_for_each_skill(self):
        return [a[1] - a[0] for a in self.skill_time_as_tuple_list]


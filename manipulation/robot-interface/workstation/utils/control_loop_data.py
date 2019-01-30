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
        self.robot_data_by_skill_info_dict = self.get_robot_data_for_skill_info(
                data_as_ord_dict, self.skill_time_as_tuple_list)

    def get_times_for_separate_skills(self, data_as_ord_dict):
        time_arr = data_as_ord_dict['time']
        start_times, _ = np.where(time_arr <= 0.0)
        end_times = [start_times[i] for i in range(1, len(start_times))]
        end_times.append(time_arr.shape[0])
        return list(zip(start_times, end_times))

    def get_duration_for_each_skill(self):
        return [a[1] - a[0] for a in self.skill_time_as_tuple_list]

    def set_skill_list(self, skill_list):
        assert len(skill_list) == len(self.skill_time_as_tuple_list), \
                "Skill list len does not match skill time as tuple"
        self.skill_list = skill_list

    def get_robot_data_for_skill_info(self, data_as_ord_dict,
                                      skill_time_as_tuple_list):
        if data_as_ord_dict.get('skill_info') is None:
            return None
        skill_info_list = data_as_ord_dict['skill_info']
        assert len(skill_info_list) == len(skill_time_as_tuple_list), \
                "Number of skills and their duration do not match"
        robot_data_by_skill_info_dict = {}
        for i, skill_info in enumerate(skill_info_list):
            skill_duration = skill_time_as_tuple_list[i]
            start_idx, end_idx = skill_duration[0], skill_duration[1]
            robot_data_by_skill_info_dict[skill_info] = {}
            for data_key, data_arr in data_as_ord_dict.items():
                if data_key == 'skill_info':
                    continue
                skill_data_arr = data_arr[start_idx:end_idx, ...]
                robot_data_by_skill_info_dict[skill_info][data_key] = \
                        skill_data_arr

        return robot_data_by_skill_info_dict


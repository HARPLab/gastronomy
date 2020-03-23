import numpy as np

import argparse
import csv
import h5py
import pdb
import pprint

from collections import OrderedDict

class ControlLoopData(object):
    def __init__(self, data_as_ord_dict):
        self.data_as_ord_dict = data_as_ord_dict
        self.skill_time_as_tuple_list = self.get_times_for_separate_skills(
                data_as_ord_dict)
        self.robot_data_by_skill_info_dict = self.get_robot_data_for_skill_info(
                data_as_ord_dict, self.skill_time_as_tuple_list)

    def get_times_for_separate_skills(self, data_as_ord_dict):
        time_arr = data_as_ord_dict['time_since_skill_started']
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

class ControlLoopDataHDF5(object):
    def __init__(self, robot_state_by_desc_dict, skill_info_dict):
        self._robot_state_by_desc_dict = robot_state_by_desc_dict
        self._skill_info_dict = skill_info_dict
        
    def get_robot_pose_for_skills(self):
        pose_for_skill_dict = {}
        for skill_desc in self._skill_info_dict['skill_sequence']:
            if skill_desc in self._robot_state_by_desc_dict:
                robot_skill_dict = self._robot_state_by_desc_dict[skill_desc]
                if 'pose' in robot_skill_dict:
                    pose_for_skill_dict[skill_desc] = robot_skill_dict['pose']
        return pose_for_skill_dict
    
    def get_skill_sequence_for_demonstration(self):
        return self._skill_info_dict['skill_sequence']

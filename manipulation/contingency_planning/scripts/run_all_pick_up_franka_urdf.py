import argparse
import glob
import subprocess
import random
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf_starting_number', '-u', type=int, default=26)
    args = parser.parse_args()

    gpu_nums = [0]
    num_gpus = len(gpu_nums)
    num_processes_per_gpu = 3
    processes = []

    for i in range(num_gpus):
        for j in range(num_processes_per_gpu):
            command = "python scripts/run_pick_up_franka_urdf.py -g " + str(gpu_nums[i]) + " -u "+str(args.urdf_starting_number + i*num_processes_per_gpu + j)
            processes.append(subprocess.Popen(command, shell=True))

    exit_codes = [p.wait() for p in processes]

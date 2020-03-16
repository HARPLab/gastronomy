import os
import argparse
import logging
from pickle import dump
import csv
import numpy as np
from tqdm import trange


def parse_txt_file(path):
    data = []
    headers = []
    header_idxs = [0]
    cur_skill_data = {}
    cur_skill_state_dict = {'skill_id': -1, 'skill_desc': None}

    n_rows = 0
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for _ in csv_reader:
            n_rows += 1 

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for n_row in trange(n_rows):
            row = next(csv_reader)
            if n_row == 0:
                for raw in row:
                    p_start, p_end = raw.find('('), raw.find(')')
                    header = raw[:p_start]
                    size = int(raw[p_start + 1:p_end])

                    headers.append(header)
                    header_idxs.append(header_idxs[-1] + size)
                cur_skill_data = {header: [] for header in headers}
            else:
                if 'info' in row[0]:
                    for key, val in cur_skill_data.items():
                        cur_skill_data[key] = np.array(val)
                    cur_skill_data.update(cur_skill_state_dict)
                    data.append(cur_skill_data)

                    cur_skill_data = {header: [] for header in headers}
                    cur_skill_state_dict = {
                        'skill_id': int(row[0][-row[0][::-1].find(':') + 1:]), 
                        'skill_desc': row[3][row[3].find(':')+2:]
                    }
                else:
                    row_data = np.array(row, dtype='float')
                    for n_header, header in enumerate(headers):
                        start_idx, end_idx = header_idxs[n_header], header_idxs[n_header + 1]
                        cur_skill_data[header].append(row_data[start_idx : end_idx])
    
    return data


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str)
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.dirname(args.input)

    output_path = os.path.join(output_dir, '{}.pkl'.format(os.path.basename(args.input)[:-4]))
    logging.info('Parsing {}'.format(args.input))
    logging.info('Writing to {}'.format(output_path))

    data = parse_txt_file(args.input)
    with open(output_path, 'wb') as f:
        dump(data, f)

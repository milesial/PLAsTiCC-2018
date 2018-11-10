from os.path import join, dirname
from tqdm import tqdm
import csv

from utils.constants import NUM_TEST_ROWS


def split_test_set(csv_path, num_files=8):
    """Splits a large csv file into N, not splitting an object across two files"""
    chunk_size = int(NUM_TEST_ROWS / num_files)
    print(f'\nSplitting {csv_path} into {num_files} csv. Approx {chunk_size} rows per file\n')

    out_files = [open(join(dirname(csv_path), f'test_set_{i}.csv'), 'w') for i in range(num_files)]
    writers = [csv.writer(f, lineterminator='\n') for f in out_files]
    current_writer = 0
    last_oid = None

    with open(csv_path, 'r') as csvfile, tqdm(total=NUM_TEST_ROWS, unit_scale=True) as pbar:
        reader = csv.reader(csvfile)
        pbar.set_postfix_str(f'({current_writer+1} out of {num_files})')
        for i_row, row in enumerate(reader):
            if i_row == chunk_size * (current_writer + 1):
                last_oid = row[0]

            if i_row > chunk_size * (current_writer + 1) and row[0] != last_oid:
                out_files[current_writer].close()
                current_writer += 1
                pbar.set_postfix_str(f'({current_writer+1} out of {num_files})')

            writers[current_writer].writerow(row)
            pbar.update()

        out_files[-1].close()


if __name__ == '__main__':
    split_test_set('../../data/test_set.csv', num_files=8)

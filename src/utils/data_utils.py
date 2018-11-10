""" Utilities to load data """

import pandas as pd
import numpy as np
from os.path import dirname, join

from utils.constants import CLASSES

DATA_DIR = '../../data'
DATA_DIR = join(dirname(__file__), DATA_DIR)

CHECKPOINTS_DIR = '../../checkpoints'
CHECKPOINTS_DIR = join(dirname(__file__), CHECKPOINTS_DIR)


def get_train_set_metadata() -> pd.DataFrame:
    path = join(DATA_DIR, 'training_set_metadata.csv')
    return pd.read_csv(path).astype(np.float32)


def get_train_set() -> pd.DataFrame:
    path = join(DATA_DIR, 'training_set.csv')
    return pd.read_csv(path).astype(np.float32)


def get_test_set_metadata() -> pd.DataFrame:
    path = join(DATA_DIR, 'test_set_metadata.csv')
    return pd.read_csv(path).astype(np.float32)


def get_test_set_iterator(chunk_size=10000):
    path = join(DATA_DIR, 'test_set.csv')
    return pd.read_csv(path, chunksize=chunk_size).astype(np.float32)


def get_train_objects():
    df = get_train_set()
    df_meta = get_train_set_metadata()
    for oid, o_frame in df.groupby('object_id'):
        meta_row = df_meta[df_meta['object_id'] == oid]
        meta_row = meta_row.drop('object_id', axis=1).squeeze()

        passband_ts = [frame.drop(['object_id', 'passband'], axis=1)
                       for _, frame in o_frame.groupby('passband')]

        yield passband_ts, meta_row


def target_to_one_hot(target):
    one_hot = np.zeros(len(CLASSES), np.int8)
    one_hot[CLASSES.index(target)] = 1
    return one_hot


if __name__ == '__main__':
    for i in get_train_objects():
        print(i)

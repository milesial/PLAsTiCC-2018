""" Utilities to load data """

import pandas as pd
from os.path import dirname, join

DATA_DIR = '../../data'
DATA_DIR = join(dirname(__file__), DATA_DIR)

CHECKPOINTS_DIR = '../../checkpoints'
CHECKPOINTS_DIR = join(dirname(__file__), CHECKPOINTS_DIR)


def get_train_set_metadata() -> pd.DataFrame:
    path = join(DATA_DIR, 'training_set_metadata.csv')
    return pd.read_csv(path)


def get_train_set() -> pd.DataFrame:
    path = join(DATA_DIR, 'training_set.csv')
    return pd.read_csv(path)


def get_test_set_metadata() -> pd.DataFrame:
    path = join(DATA_DIR, 'test_set_metadata.csv')
    return pd.read_csv(path)


def get_test_set_iterator(chunk_size=10000):
    path = join(DATA_DIR, 'test_set.csv')
    return pd.read_csv(path, chunksize=chunk_size)


def get_train_objects():
    df = get_train_set()
    for oid, o_frame in df.groupby('object_id'):
        yield [frame.drop(['object_id', 'passband'], axis=1)
                for _, frame in o_frame.groupby('passband')]


if __name__ == '__main__':
    for i in get_train_objects():
        print(len(i))
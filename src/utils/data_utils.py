""" Utilities to load data """

import pandas as pd
import numpy as np
from os.path import dirname, join

from utils.constants import CLASSES

DATA_DIR = '../../data'
DATA_DIR = join(dirname(__file__), DATA_DIR)


def read_csv_and_fill_na(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype=np.float32).fillna(0)


def get_train_set_metadata() -> pd.DataFrame:
    path = join(DATA_DIR, 'training_set_metadata.csv')
    return read_csv_and_fill_na(path)


def get_train_set() -> pd.DataFrame:
    path = join(DATA_DIR, 'training_set.csv')
    return read_csv_and_fill_na(path)


def get_test_set_metadata() -> pd.DataFrame:
    path = join(DATA_DIR, 'test_set_metadata.csv')
    return read_csv_and_fill_na(path)


def get_test_set(csv_id) -> pd.DataFrame:
    path = join(DATA_DIR, f'test_set_{csv_id}.csv')
    return read_csv_and_fill_na(path)


def get_objects(df, df_meta):
    """
    From the time series df and the metadata df, yields objects one by one with:

      - A time series for each passband as dataframes
      - The metadata row as a one-row dataframe
    """
    for oid, o_frame in df.groupby('object_id'):
        meta_row = df_meta[df_meta['object_id'] == oid]
        meta_row = meta_row.drop('object_id', axis=1).squeeze()

        passband_ts = [frame.drop(['object_id', 'passband'], axis=1)
                       for _, frame in o_frame.groupby('passband')]

        yield passband_ts, meta_row


def get_test_objects(csv_id):
    df = get_test_set(csv_id)
    df_meta = get_test_set_metadata()
    return get_objects(df, df_meta)


def get_train_objects():
    df = get_train_set()
    df_meta = get_train_set_metadata()
    return get_objects(df, df_meta)


def target_to_one_hot(target):
    one_hot = np.zeros(len(CLASSES), np.int8)
    one_hot[CLASSES.index(target)] = 1
    return one_hot

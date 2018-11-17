import numpy as np
from cesium.featurize import featurize_time_series

from utils.data_utils import get_train_objects


def get_features(time_series):
    times = [df['mjd'].tolist() for df in time_series]
    values = [df['flux'].tolist() for df in time_series]
    errors = [df['flux_err'].tolist() for df in time_series]

    features = featurize_time_series(
        times, values, errors,
        features_to_use=[
            'amplitude',
            'median',
            'median_absolute_deviation',
            'minimum',
            'maximum',
            'all_times_nhist_numpeaks'
        ]
    )
    return features.values


def featurize_train_object(time_series, metadata):
    features = get_features(time_series)
    features_vector = np.hstack(features)
    full_vector = np.hstack([features_vector, metadata.drop('target').values])
    full_vector[np.isnan(full_vector)] = 0
    return full_vector, metadata['target']


def featurize_test_object(time_series, metadata):
    features = get_features(time_series)
    features_vector = np.hstack(features)
    full_vector = np.hstack([features_vector, metadata.values])
    full_vector[np.isnan(full_vector)] = 0
    return full_vector


def get_featurized_train_objects():
    train_set = get_train_objects()
    train_set = map(lambda o: featurize_train_object(o[1], o[2]), train_set)
    return train_set

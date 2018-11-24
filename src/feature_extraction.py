import csv
from multiprocessing import Manager
from multiprocessing import Pool

import numpy as np
from cesium.featurize import featurize_time_series
from tqdm import tqdm

from utils.constants import METADATA_COLUMNS
from utils.constants import NUM_PASSBANDS
from utils.constants import NUM_TEST_EXAMPLES
from utils.constants import NUM_TRAIN_EXAMPLES
from utils.data_utils import get_num_test_shards
from utils.data_utils import get_test_objects
from utils.data_utils import get_train_objects

FEATURES = [
    'amplitude',
    'median',
    'median_absolute_deviation',
    'minimum',
    'maximum',
    'all_times_nhist_numpeaks',
    'total_time',
    'flux_percentile_ratio_mid20',
    'flux_percentile_ratio_mid50',
    'flux_percentile_ratio_mid80',
    'percent_close_to_median',
    'percent_difference_flux_percentile',
    'skew',
    'std',
    'freq1_amplitude1',
    'freq1_amplitude2',
    'freq1_amplitude3',
    'freq1_freq',
    'freq1_lambda',
    'freq1_rel_phase2',
    'freq1_rel_phase3',
    'freq1_rel_phase4',
    'freq1_signif',
    'freq2_freq',
    'freq3_freq',
    'freq_y_offset',
    'fold2P_slope_10percentile',
    'fold2P_slope_90percentile',
    'freq1_freq',
    'freq1_lambda',
    'freq_varrat',
    'linear_trend',
    'freq_signif_ratio_21',
    'freq_signif_ratio_31',
    'stetson_j',
    'stetson_k'
]


def get_features(time_series):
    times = [df['mjd'].tolist() for df in time_series]
    values = [df['flux'].tolist() for df in time_series]
    errors = [df['flux_err'].tolist() for df in time_series]

    features = featurize_time_series(
        times, values, errors,
        features_to_use=FEATURES
    )
    return features.values


def featurize_train_object(time_series, metadata):
    features = get_features(time_series)
    features_vector = np.hstack(features)
    full_vector = np.hstack([features_vector, metadata.drop('target').values])
    full_vector[np.isnan(full_vector)] = 0
    full_vector[np.isinf(full_vector)] = 0
    return full_vector, np.asarray(metadata['target'])


def featurize_test_object(time_series, metadata):
    features = get_features(time_series)
    features_vector = np.hstack(features)
    full_vector = np.hstack([features_vector, metadata.values])
    full_vector[np.isnan(full_vector)] = 0
    full_vector[not np.isfinite(full_vector)] = 0
    return full_vector


def _featurize_train_fn(object):
    return featurize_train_object(object[1], object[2])


def get_featurized_train_objects(n_workers, verbose=False):
    if verbose:
        print('\nLoading train dataset...')

    train_set = []
    for o in tqdm(get_train_objects(), total=NUM_TRAIN_EXAMPLES):
        train_set.append(o)

    if verbose:
        print('\nComputing features...')
        bar = tqdm(total=NUM_TRAIN_EXAMPLES)

    chunksize = int(NUM_TRAIN_EXAMPLES / n_workers)
    with Pool(n_workers) as pool:
        for out in pool.imap(_featurize_train_fn, train_set, chunksize):
            if verbose:
                bar.update()
            yield out


def _featurize_test_shard(queue, shard_id):
    for oid, ts, meta in get_test_objects(shard_id):
        feature = featurize_test_object(ts, meta)
        queue.put([oid, *feature])


def featurize_and_save_test_set(n_workers, out_csv):
    pbar = tqdm(total=NUM_TEST_EXAMPLES)
    features = [f'band{b}_{f}' for b in range(NUM_PASSBANDS) for f in FEATURES]
    header = ['object_id', *features, *METADATA_COLUMNS[:-1]]
    n_shards = get_num_test_shards()
    chunksize = int(n_shards / n_workers)
    if chunksize < 1:
        chunksize = 1

    m = Manager()
    q = m.Queue()
    with Pool(n_workers) as pool:
        r = pool.starmap_async(_featurize_test_shard, [(q, i) for i in range(n_shards)], chunksize=1)

        with open(out_csv, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(header)
            while True:
                item = q.get(block=True)
                writer.writerow(item)
                pbar.update(1)

        r.wait()


def featurize_and_save_train_set(n_workers, out_csv):
    features = [f'band{b}_{f}' for b in range(NUM_PASSBANDS) for f in FEATURES]
    header = ['object_id', *features, *METADATA_COLUMNS]

    with open(out_csv, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)

        for feat, label in get_featurized_train_objects(n_workers, verbose=True):
            writer.writerow([oid, *feat, label])


if __name__ == '__main__':
    featurize_and_save_test_set(4, '../data/test_set_features.csv')

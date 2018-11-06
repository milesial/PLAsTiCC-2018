from cesium.featurize import featurize_time_series


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
    return features

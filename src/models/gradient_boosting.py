from sklearn.ensemble import GradientBoostingClassifier


def get_model(lr=0.1, verbose=False):
    return GradientBoostingClassifier(
        loss='deviance',  # logistic regression
        learning_rate=lr,
        n_estimators=2000,
        subsample=0.9,
        max_depth=3,
        max_features='sqrt',
        verbose=verbose,
        n_iter_no_change=10
    )


if __name__ == '__main__':
    import numpy as np
    from sklearn.externals import joblib
    from os.path import join, dirname, exists

    from feature_extraction import get_featurized_train_objects
    from feature_extraction import featurize_test_object

    from submission.prediction import predict_test_probs

    model_nickname = 'gradient_boosting_v1'

    checkpoint_path = join(dirname(__file__), '../../checkpoints')
    checkpoint_path = join(checkpoint_path, model_nickname + '.joblib')

    if not exists(checkpoint_path):
        print('Building dataset')
        X = []
        y = []
        for ts, label in get_featurized_train_objects(n_workers=10, pbar=True):
            X.append(ts)
            y.append(label)
        print('Finished building dataset')

        X = np.vstack(X)
        y = np.array(y, np.int8)

        model = get_model(lr=0.1, verbose=True)
        model.fit(X, y)

        print('train_accuracy', model.score(X, y))

        # free some memory
        del X, y

        joblib.dump(model, checkpoint_path)
    else:
        model = joblib.load(checkpoint_path)

    predict_test_probs(featurize_test_object, model.predict_proba, model_nickname + '.csv', 10)

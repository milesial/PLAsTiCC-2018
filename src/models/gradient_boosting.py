from scipy.stats import randint as sp_randint
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV


def get_model(lr=0.1, verbose=False):
    return GradientBoostingClassifier(
        loss='deviance',  # logistic regression
        learning_rate=lr,
        n_estimators=2000,
        subsample=0.5,
        max_depth=5,
        validation_fraction=0.2,
        max_features=0.7,
        verbose=verbose,
        n_iter_no_change=2
    )


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


if __name__ == '__main__':
    import numpy as np
    from sklearn.externals import joblib
    from os.path import join, dirname, exists

    from feature_extraction import get_featurized_train_objects

    model_nickname = 'gradient_boosting_v2_features'

    checkpoint_path = join(dirname(__file__), '../../checkpoints')
    checkpoint_path = join(checkpoint_path, model_nickname + '.joblib')

    if not exists(checkpoint_path):
        print('Building dataset')
        X = []
        y = []
        for feat, label in get_featurized_train_objects(n_workers=10, verbose=True):
            X.append(feat)
            y.append(label)
        print('Finished building dataset')

        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        # X_train = np.vstack(X_train)
        # y_train = np.array(y_train, np.int8)
        # X_val = np.vstack(X_val)
        # y_val = np.array(y_val, np.int8)

        X = np.vstack(X)
        y = np.array(y, np.int8)

        param_dist = {"max_depth": sp_randint(2, 10),
                      "max_features": [0.5, 1.0],
                      "subsample": [0.5, 1.0]}

        model = GradientBoostingClassifier(
            loss='deviance',  # logistic regression
            learning_rate=0.1,
            n_estimators=2000,
            validation_fraction=0.2,
            verbose=True,
            n_iter_no_change=2
        )

        random_search = RandomizedSearchCV(model,
                                           param_distributions=param_dist,
                                           n_iter=10,
                                           n_jobs=4,
                                           cv=10)

        random_search.fit(X, y)
        report(random_search.cv_results_)

        # model.fit(X_train, y_train)

        # print('train_accuracy', model.score(X_train, y_train))
        # print('val_accuracy', model.score(X_val, y_val))

        joblib.dump(model, checkpoint_path)
    else:
        model = joblib.load(checkpoint_path)

    # predict_test_probs(featurize_test_object, model.predict_proba, model_nickname + '.csv', 10)

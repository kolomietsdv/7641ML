import time

import sklearn.metrics
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sps
import itertools


def time_it(f, *args, **kwargs):
    time_start = time.time()
    res = f(*args, **kwargs) 
    duration = time.time() - time_start
    return res, duration


def custom_scorer_factory(f, cl):
    def custom_metric(estimator, x, y):
        preds = estimator.predict(x)
        return f(np.where(y == cl, 1, 0),
                 np.where(preds == cl, 1, 0))
    return custom_metric


def build_scorers(class_labels):
    scorer_dict = {
        f"precision_{cl}": custom_scorer_factory(sklearn.metrics.precision_score, cl)
        for cl in sorted(class_labels)
    }

    scorer_dict.update({
        f"recall_{cl}": custom_scorer_factory(sklearn.metrics.recall_score, cl)
        for cl in sorted(class_labels)
    })

    scorer_dict['accuracy'] = lambda estimator, x, y: sklearn.metrics.accuracy_score(y, estimator.predict(x))
    return scorer_dict


def brute_force_parameters(par_name, par_values, estimator_kwargs, x_train, y_train, estimator=None):
    np.random.seed(13)

    scorers = build_scorers(np.unique(y_train))
    
    metrics = {}

    if not estimator:
        estimator = DecisionTreeClassifier

    if type(par_name) == list:
        total = len(list(itertools.product(*par_values)))
        for i, par_set in enumerate(itertools.product(*par_values), start=1):
            print(f'\r{i}/{total}', end='')
            estimator_kwargs.update(dict(zip(par_name, par_set)))
            clf = estimator(**estimator_kwargs)
            metrics[par_set] = cross_validate(clf, X=x_train, y=y_train, cv=3, n_jobs=1, scoring=scorers,
                                              return_train_score=True)
    else:
        total = len(par_values)
        for i, par_value in enumerate(par_values, start=1):
            print(f'\r{i}/{total}', end='')
            estimator_kwargs[par_name] = par_value
            clf = estimator(**estimator_kwargs)
            metrics[par_value] = cross_validate(clf, X=x_train, y=y_train, cv=3, n_jobs=1, scoring=scorers,
                                                return_train_score=True)

    metrics = dict(
        map(lambda x: (x[0],
                       dict(map(lambda y: (y[0], np.mean(y[1]).round(4)), x[1].items()))),
            metrics.items())
    )
    return metrics


def extract_baselines(metric_object):
    baselines = {}
    for x in [
        'accuracy',
        'precision_-1', 'precision_0', 'precision_1',
        'recall_-1', 'recall_0', 'recall_1']:
        baselines[x] = metric_object[f'test_{x}']
    return baselines


def plot_metrics(metrics_dict, class_names, baselines=None):

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(14, 10)

    x_values = sorted(metrics_dict.keys())

    axes[0][0].plot(x_values,
                    [metrics_dict[x]['test_accuracy'] for x in x_values], color='red',
                    label='accuracy')
    axes[0][0].set_title('accuracy')

    axes[0][0].plot(x_values,
                    [metrics_dict[x]['train_accuracy'] for x in x_values], color='orange',
                    label='train_accuracy')
    # axes[0][0].set_title('accuracy')

    if baselines:
        if 'accuracy' in baselines:
            axes[0][0].axhline(baselines['accuracy'], alpha=0.5, label='baseline',
                               linestyle='--')
            axes[0][0].legend()

    axes[0][1].plot(x_values,
                    [metrics_dict[x]['fit_time'] for x in x_values], color='blue')
    axes[0][1].set_title('duration')

    colors = ['orange', 'royalblue', 'salmon']

    for i, cl in enumerate(class_names):
        axes[1][0].plot(x_values,
                        [metrics_dict[x][f'test_precision_{cl}'] for x in x_values],
                        color=colors[i], label=class_names[i])
        if baselines:
            if f'precision_{cl}' in baselines:
                axes[1][0].axhline(baselines[f'precision_{cl}'], color=colors[i],
                                   alpha=0.5, label=f'baseline_precision_{cl}',
                                   linestyle='--')
                axes[1][0].legend()

    axes[1][0].legend()
    axes[1][0].set_title('class precision')

    for i, cl in enumerate(class_names):
        axes[1][1].plot(x_values,
                        [metrics_dict[x][f'test_recall_{cl}'] for x in x_values],
                        color=colors[i], label=class_names[i])
        if baselines:
            if f'recall_{cl}' in baselines:
                axes[1][1].axhline(baselines[f'recall_{cl}'], color=colors[i],
                                   alpha=0.5, label=f'baseline_recall_{cl}',
                                   linestyle='--')
                axes[1][1].legend()

    axes[1][1].legend()
    axes[1][1].set_title('class recall')
    fig.suptitle('Max depth effect', fontsize=15)
    plt.tight_layout()
    return fig, axes


def encode_input_data(df, trained_ohes=None):
    df = df.copy()
    if not trained_ohes:
        trained_ohes = {}
    y = df['y'].values
    #     y = df['y'].apply(lambda x: 1 if x == 'yes' else 0).values
    del df['y']

    encoded = []
    is_col_object = df.dtypes == 'object'
    for col, is_object in zip(df.dtypes.index, is_col_object):
        if not is_object:
            encoded.append(df[col].values.reshape(-1, 1))
        else:
            if col not in trained_ohes:
                ohe = OneHotEncoder(handle_unknown='ignore')
                trained_ohes[col] = ohe

            ohe_encoded = trained_ohes[col].fit_transform(df[col].values.reshape(-1, 1))
            encoded.append(ohe_encoded)
    #         print(encoded[-1].shape)
    #     X = np.hstack(encoded)
    X = sps.hstack(encoded)
    return X, y, trained_ohes


def plot_heatmap(metric_dict, inner_dicts=False, colnames=['depth', 'n_neurons', 'acc']):
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)

    if inner_dicts:
        results_df = pd.DataFrame(
            list(
                map(
                lambda x: (round(x[0][0], 5), x[0][1], x[1]['test_accuracy']),
                metric_dict.items()
                )
            ), columns=colnames
        )
    else:
        results_df = pd.DataFrame(
            list(
                map(
                lambda x: (x[0][0], x[0][1], x[1][-1][-1]),
                metric_dict.items()
                )
            ), columns=colnames
        )


    sns.heatmap(
        results_df.pivot(index=colnames[0], columns=colnames[1], values=colnames[2]),
        annot=True, ax=ax
    )
    ax.set_title('Accuracy heatmap',)
    return fig, ax
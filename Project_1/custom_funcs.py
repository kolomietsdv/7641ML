import time 
import sklearn.metrics
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import numpy as np


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


def brute_force_parameters(par_name, par_values, estimator_kwargs, x_train, y_train):
    np.random.seed(13)

    scorers = build_scorers(np.unique(y_train))
    
    metrics = {}

    for par_value in par_values:
        estimator_kwargs[par_name] = par_value
        dt_clf = DecisionTreeClassifier(**estimator_kwargs)
        metrics[par_value] = cross_validate(dt_clf, X=x_train, y=y_train, cv=3, n_jobs=1, scoring=scorers)

    metrics = dict(
        map(lambda x: (x[0],
                       dict(map(lambda y: (y[0], np.mean(y[1]).round(4)), x[1].items()))),
            metrics.items())
    )
    return metrics


def plot_metrics(metrics_dict, class_names):

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(14, 10)

    x_values = sorted(metrics_dict.keys())

    axes[0][0].plot(x_values,
                    [metrics_dict[x]['test_accuracy'] for x in x_values], color='red')
    axes[0][0].set_title('accuracy')

    axes[0][1].plot(x_values,
                    [metrics_dict[x]['fit_time'] for x in x_values], color='blue')
    axes[0][1].set_title('duration')

    colors = ['orange', 'royalblue', 'salmon']

    for i, cl in enumerate(class_names):
        axes[1][0].plot(x_values,
                        [metrics_dict[x][f'test_precision_{cl}'] for x in x_values],
                        color=colors[i], label=class_names[i])
    axes[1][0].legend()
    axes[1][0].set_title('class precision')

    for i, cl in enumerate(class_names):
        axes[1][1].plot(x_values,
                        [metrics_dict[x][f'test_recall_{cl}'] for x in x_values],
                        color=colors[i], label=class_names[i])
    axes[1][1].legend()
    axes[1][1].set_title('class recall')
    fig.suptitle('Max depth effect', fontsize=15)
    plt.tight_layout()
    return fig, axes

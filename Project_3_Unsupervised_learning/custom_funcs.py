import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import kurtosis
from tqdm.notebook import tqdm
import numpy as np
import sklearn.metrics
from sklearn.model_selection import KFold
import sklearn.random_projection
from IPython.display import display
from yellowbrick.cluster.elbow import distortion_score


def time_it(f, *args, **kwargs):
    time_start = time.time()
    res = f(*args, **kwargs)
    duration = time.time() - time_start
    return res, duration


def read_data(name):
    save_path = './data/preprocessed_splits/'
    data = {}
    for split_name in ['X_train', 'X_test', 'y_train', 'y_test']:
        data[split_name] = pd.read_parquet(
            os.path.join(save_path, name, f'{split_name}.snappy.parquet')
        ).reset_index(drop=True)
    return data


def build_elbow_data(model, par_net, df, metrics):
    res = {}
    for par_name, par_set in par_net.items():
        print(f'Par: {par_name}')
        len_parset = len(par_set)
        for num, par_val in enumerate(par_set):
            res[par_val] = {}
            print(f'\rValue: {par_val}, Progress: {num}/{len_parset}.', end='')
            res[par_val] = {}
            model.set_params(**{par_name: par_val})
            # model.fit(df)
            _, duration = time_it(model.fit, df)
            preds = model.predict(df)
            res[par_val]['preds'] = preds
            res[par_val]['fit_time'] = duration

            for m_name, m in metrics.items():
                score = m(df, preds)
                res[par_val][m_name] = score
    return res


def plot_elbow(elbow_data):
    axes = pd.DataFrame(elbow_data).T.drop(columns=['preds']).plot(kind='line', subplots=True, figsize=(8, 12))
    return axes


def plot_double_elbow(elbow_datas):
    elbow_datas_items = [x for x in elbow_datas.items()]

    j = pd.DataFrame(elbow_datas_items[0][1]).T.join(pd.DataFrame(elbow_datas_items[1][1]).T,
                                                     lsuffix=f'_{elbow_datas_items[0][0]}',
                                                     rsuffix=f'_{elbow_datas_items[1][0]}')

    fig, axes = plt.subplots(3, 2)
    fig.set_size_inches(15, 10)

    num = 0

    for color, f in zip(['tab:green', 'tab:red', 'tab:blue'], ['distortion', 'silhouette', 'fit_time']):
        type = j[[x for x in j.columns if x.startswith(f)]].values
        min, max = type.min(), type.max()
        for alg_name in elbow_datas.keys():
            cur_ax = axes[num // 2][num % 2]
            colname = f'{f}_{alg_name}'
            j[colname].plot(kind='line', ax=cur_ax, color=color)

            if f == 'silhouette':
                cur_ax.axhline(j[colname].max(), linestyle='--', alpha=0.5, label='max', color=color)
            cur_ax.set_title(colname)
            cur_ax.set_xticks(j.index)
            cur_ax.set_ylim(min - max * 0.1, max + max * 0.1)
            num += 1
    return fig, axes


def plot_tsne_clustered(tsne_repr, elbow_data, ns, alg_name, par_name, palette='tab10'):
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches((16, 6))

    for num, n in enumerate(ns):
        cur_ax = axes[num]
        sns.scatterplot(x=tsne_repr[:, 0], y=tsne_repr[:, 1], s=0.3, ax=cur_ax,
                        hue=elbow_data[n]['preds'], palette=palette, size=0.1)
        cur_ax.set_title(f'{alg_name.capitalize()}, {par_name}: {n}.')
        cur_ax.legend(fontsize=4)
    return fig, axes


def build_ica_data(df):
    results = {}
    for n in tqdm(range(1, df.shape[1] + 1)):
        ica = FastICA(n_components=n, random_state=13, max_iter=15000)
        results[n] = {}
        transformed_df = ica.fit_transform(df)
        kurtosisis = kurtosis(transformed_df)
        results[n]['transformed_df'] = transformed_df
        results[n]['kurtosisis'] = kurtosisis
        results[n]['model'] = ica
    return results


def component_extremums(df, components, n_samples=3, first_n_components=3):
    results = {}
    for comp_num in range(first_n_components):
        results[comp_num] = {}

        transformed_data = pd.DataFrame(
            df.values.dot(components[comp_num].T),
            columns=['x']
        )

        sorted_index = transformed_data.sort_values('x').index
        results[comp_num]['one_side'] = df.iloc[sorted_index[:n_samples]]
        results[comp_num]['opposite_side'] = df.iloc[sorted_index[-n_samples:]]

        results[comp_num]['feature_distance'] = (
                results[comp_num]['one_side'].mean() - results[comp_num]['opposite_side'].mean()
        ).abs().sort_values(ascending=False)

    return results


def compute_reconstruction_error(df, fitted_pca_model=None):
    results = {}
    if not fitted_pca_model:
        fitted_pca_model = PCA(random_state=13)
        fitted_pca_model.fit(df)

    for n in range(1, len(fitted_pca_model.components_) + 1):
        first_n_comps = fitted_pca_model.components_[:n, :].T
        transformed_data = df.values.dot(first_n_comps)
        inverse_data = np.linalg.pinv(first_n_comps)
        reconstructed_data = transformed_data.dot(inverse_data)
        results[n] = sklearn.metrics.mean_squared_error(y_true=df.values, y_pred=reconstructed_data)
    return results


def compute_and_plot_rgp_reconstruction_error(dfs):
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 4)
    colors = {'wine': 'tab:orange', 'bike': 'tab:blue'}

    for num, (name, df) in enumerate(dfs.items()):
        rgp = sklearn.random_projection.GaussianRandomProjection(df.shape[1])
        rgp.fit(df)
        pd.Series(compute_reconstruction_error(df, fitted_pca_model=rgp)).plot(
            ax=axes[num], color=colors[name]
        )
        axes[num].set_title(f'{name.capitalize()} dataset.')
        axes[num].set_ylim((0, 1))
    fig.suptitle('Reconstruction Error', fontsize=15)
    return fig, axes

    # rgp_wine = random_projection.GaussianRandomProjection(wine_data['X_train'].shape[1])
    # rgp_wine.fit(wine_data['X_train'])
    # pd.Series(cf.compute_reconstruction_error(wine_data['X_train'], fitted_pca_model=rgp_wine)).plot(ax=axes[1])


def cross_validate(X, y, model, regression=False, n_splits=3, progress=True, random_state=13):
    results = {}
    kf = KFold(n_splits=n_splits, random_state=random_state)

    results['scores'] = []
    results['duration'] = []
    if regression:
        results['mse'] = []
    else:
        results['accuracy'] = []

    if progress:
        iterator = tqdm(kf.split(X), total=n_splits)
    else:
        iterator = kf.split(X)

    for train_index, test_index in iterator:
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        _, duration = time_it(model.fit, X_train, y_train)
        results['duration'].append(duration)

        if regression:
            cur_scores = model.predict(X_valid)
            results['scores'].append(cur_scores)
            results['mse'].append(sklearn.metrics.mean_squared_error(y_true=y_valid, y_pred=cur_scores))
        else:
            cur_scores = model.predict(X_valid)
            results['scores'].append(cur_scores)
            results['accuracy'].append(sklearn.metrics.accuracy_score(y_true=y_valid, y_pred=cur_scores))

    results['mean_duration'] = np.mean(results['duration'])
    if regression:
        results['mean_mse'] = np.mean(results['mse'])
    else:
        results['mean_accuracy'] = np.mean(results['accuracy'])

    return results


def single_wrapping_experiment(df, y, model, transformer_class, transformer_kwargs, regression=False, n_splits=3,
                               tries=1,
                               random_state=13):
    results = []

    for i in tqdm(range(tries)):
        transformer = transformer_class(random_state=random_state + i, **transformer_kwargs)
        transformed_df = transformer.fit_transform(df)
        cur_results = cross_validate(transformed_df, y, model, regression=regression, n_splits=n_splits)
        results.append(cur_results)
    return results


def multi_wrapping_experiment(par_grid, df, y, model, transformer_class, transformer_kwargs, regression=False,
                              n_splits=3, tries=1,
                              random_state=13):
    results = {}
    par_name, par_range = list(par_grid.items())[0]
    for par_val in tqdm(par_range):
        transformer_kwargs[par_name] = par_val
        results[par_val] = single_wrapping_experiment(
            df=df,
            y=y,
            model=model,
            transformer_class=transformer_class, transformer_kwargs=transformer_kwargs,
            regression=regression, n_splits=n_splits, tries=tries, random_state=random_state
        )
    return results


def plot_rgp_exp_results(multi_exp_results, original_data_performance, regression=False):
    if regression:
        color = 'tab:blue'
        metric_name = 'mean_mse'
    else:
        metric_name = 'mean_accuracy'
        color = 'tab:orange'

    trial_means = pd.DataFrame(multi_exp_results).applymap(lambda x: x[metric_name]).T
    trial_durs = pd.DataFrame(multi_exp_results).applymap(lambda x: x['mean_duration']).T

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)

    # Means
    if regression:
        trial_means.min(axis=1).plot(ax=axes[0], color=color, label='random_projection_data')
    else:
        trial_means.max(axis=1).plot(ax=axes[0], color=color, label='random_projection_data')

    axes[0].fill_between(
        trial_means.index, trial_means.min(axis=1), trial_means.max(axis=1), color=color, alpha=.1
    )
    axes[0].set_title(metric_name.capitalize().replace('_', ' '))

    # Durs
    if regression:
        trial_durs.min(axis=1).plot(ax=axes[1], color=color, label='random_projection_data')
    else:
        trial_durs.max(axis=1).plot(ax=axes[1], color=color, label='random_projection_data')
    axes[1].fill_between(
        trial_durs.index, trial_durs.min(axis=1), trial_durs.max(axis=1), color=color, alpha=.1
    )
    axes[1].set_title('Mean duration')

    axes[0].axhline(
        original_data_performance[metric_name], linestyle='--', alpha=0.5, c=color,
        label='original_data'
    )
    axes[0].legend()

    axes[1].axhline(
        original_data_performance['mean_duration'], linestyle='--', alpha=0.5, c=color,
        label='original_data'
    )
    axes[1].legend()
    fig.suptitle('Neural net performance', fontsize=20)
    return fig, axes


class ForwardSearcher:
    def __init__(self, X, y, model, regression=False):
        self.model = model
        self.X = X
        self.y = y
        self.n_splits = 3
        self.best_features = []
        self.all_features = self.X.columns
        self.regression = regression
        self.each_step_results = []
        self.random_state=13

    def step(self):
        results = {}
        features_to_search = [x for x in self.all_features if x not in self.best_features]
        for feature in tqdm(features_to_search):
            cur_set = self.best_features + [feature]
            cur_feature_results = cross_validate(
                X=self.X[cur_set].values, y=self.y.values.reshape(-1), progress=False,
                model=self.model, regression=self.regression, n_splits=self.n_splits
            )
            results[feature] = cur_feature_results
        self.each_step_results.append(results)
        if self.regression:
            best_feature = min(list(results.items()), key=lambda x: x[1]['mean_mse'])[0]
        else:
            best_feature = max(list(results.items()), key=lambda x: x[1]['mean_accuracy'])[0]

        self.best_features.append(best_feature)

    def full_search(self):
        while len(self.best_features) < len(self.all_features):
            self.step()

    def plot_results(self):
        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(15, 5)

        if self.regression:
            m_name = 'mean_mse'
            color = 'tab:blue'
        else:
            m_name = 'mean_accuracy'
            color = 'tab:orange'

        best_features_n_stats = pd.DataFrame(
            {x: (self.each_step_results[num][x][m_name],
                 self.each_step_results[num][x]['mean_duration'])
             for num, x in enumerate(self.best_features)}
        ).T.round(4)

        best_features_n_stats[0].plot(ax=axes[0], color=color, label='Best found subset')
        axes[0].set_xticks(range(0, best_features_n_stats.shape[0]))
        axes[0].set_xticklabels(list(best_features_n_stats.index), rotation=90)
        axes[0].set_title(f'{m_name.capitalize().replace("_", " ")}')
        axes[0].axhline(
            best_features_n_stats[0].iloc[-1], color=color, linestyle='--', alpha=0.5, label='Full data'
        )
        axes[0].legend()

        best_features_n_stats[1].plot(ax=axes[1], color=color, label='Best found subset')
        axes[1].set_xticks(range(0, best_features_n_stats.shape[0]))
        axes[1].set_xticklabels(list(best_features_n_stats.index), rotation=90)
        axes[1].set_title('Time duration')
        axes[1].axhline(
            best_features_n_stats[1].iloc[-1], color=color, linestyle='--', alpha=0.5, label='Full data'
        )
        axes[1].legend()
        fig.suptitle('Forward Search results', fontsize=20)
        return fig, axes


def cluster_representatives(df, cluster_labels, y=None):
    if y is not None:
        df = pd.concat([df, y], axis=1)
    clusters = np.unique(cluster_labels).tolist()
    cluster_descriptors = {}
    for cl in clusters:
        cl_index = (cluster_labels == cl)
        cluster_points = df.iloc[cl_index]
        variances = cluster_points.var(axis=0).round(3).reset_index().rename(columns={'index': 'fname', 0: 'variance'})
        means = cluster_points.mean(axis=0).round(3).reset_index().rename(columns={'index': 'fname', 0: 'mean'})
        value_mass = (
            cluster_points
                .apply(value_counter, axis=0)
                .reset_index()
                .rename(columns={'index': 'fname', 0: 'value_mass'})
        )
        cur_cluster_descriptors = (
            variances
                .merge(means, on='fname', how='left')
                .merge(value_mass, on='fname', how='left')
                .sort_values('variance')
        )
        # display(variances.merge(means, on='fname', how='left'))
        cluster_descriptors[cl] = {'descr': cur_cluster_descriptors,
                                   'volume': cluster_points.shape[0]}
    return cluster_descriptors


def print_out_representatives(cluster_representatives):
    [
        (print(f'Cluster: {x[0]}, volume: {x[1]["volume"]}'),
         display(x[1]["descr"])) for x in cluster_representatives.items()
    ]


def value_counter(x):
    rounded_x = np.round(x, 2)
    value_counts = pd.value_counts(rounded_x, normalize=True)
    value_counts_rounded_sorted = value_counts.round(2).sort_values(ascending=False).head(3)
    return value_counts_rounded_sorted.to_dict()


def filter_cluster_step1(data=None, data_name='X_train_grp'):
    elbow_km = build_elbow_data(KMeans(), par_net={'n_clusters': range(2, 35, 1)},
                                df=data[data_name],
                                metrics={'distortion': distortion_score,
                                         'silhouette': sklearn.metrics.silhouette_score})

    elbow_em = build_elbow_data(GaussianMixture(), par_net={'n_components': range(2, 35, 1)},
                                df=data[data_name],
                                metrics={'distortion': distortion_score,
                                         'silhouette': sklearn.metrics.silhouette_score})
    fig, axes = plot_double_elbow({'km': elbow_km, 'em': elbow_em})
    return (elbow_km, elbow_em), (fig, axes)


def cluster_analysis_substep(ns, data, elbows, tsne_name='X_train_grp_tsne',
                             alg_name='kmeans'):
    if alg_name == 'kmeans':
        par_name = 'n_clusters'
    elif alg_name == 'em':
        par_name = 'n_components'

    fig, axes = plot_tsne_clustered(
        data[tsne_name], elbows, ns=ns, alg_name=alg_name, par_name=par_name
    )
    results = {}
    for n in ns:
        cluster_reprs = cluster_representatives(
            data['X_train'], elbows[n]['preds'], y=data['y_train_scaled']
        )
        print(f"Clustering algorithm: {alg_name}")
        print_out_representatives(cluster_reprs)
        results[n] = cluster_reprs

    return results, fig, axes


def cluster_analysis_step2(data, alg_names_elbows_nss, tsne_name):

    results = {}
    for name, (elbows, ns) in alg_names_elbows_nss.items():
        results[name] = cluster_analysis_substep(ns, data, elbows, tsne_name=tsne_name, alg_name=name)


def check_learning_influence(X_grid, y, model_class, model_kwargs, regression=False, n_splits=3):
    results = {}
    for name, X_train in tqdm(X_grid.items()):
        model = model_class(**model_kwargs)
        cur_X_results = cross_validate(
            X=X_train, y=y.values.reshape(-1), progress=False,
            model=model, regression=regression, n_splits=n_splits
        )
        results[name] = cur_X_results

    return results


def plot_learning_check_results(learning_check_results, original_data_results, regression=False):
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)

    if regression:
        m_name = 'mean_mse'
        color = 'tab:blue'
    else:
        m_name = 'mean_accuracy'
        color = 'tab:orange'

    learning_check_results = pd.DataFrame(learning_check_results).T[[m_name, 'mean_duration']]

    learning_check_results[m_name].plot(ax=axes[0], color=color, label='transformed data')
    axes[0].set_title(f'{m_name.capitalize().replace("_", " ")}')
    axes[0].axhline(
        original_data_results[m_name], color=color, linestyle='--', alpha=0.5, label='original data'
    )
    axes[0].legend()

    learning_check_results['mean_duration'].plot(ax=axes[1], color=color, label='transformed data')
    axes[1].set_title('Time duration')
    axes[1].axhline(
        original_data_results['mean_duration'], color=color, linestyle='--', alpha=0.5, label='original data'
    )
    axes[1].legend()
    fig.suptitle('Transformed data learning', fontsize=20)
    return fig, axes

def filter_learner_check(datas, model_grid):
    results = {}
    i = 1
    for df_name in ['wine', 'bike']:
        results[df_name] = {}
        for filter_name in ['pca', 'ica', 'grp', 'elim']:
            results[df_name][filter_name] = {}

            print(f'\r{i}/8. Df: {df_name}. Filter: {filter_name}.', end='')
            regression = df_name == 'bike'

            if regression:
                model_class, model_kwargs = model_grid['regression']
            else:
                model_class, model_kwargs = model_grid['classification']

            model = model_class(**model_kwargs)
            cur_X_filtered = datas[df_name][f'X_train_{filter_name}']

            if type(cur_X_filtered) == pd.DataFrame:
                cur_X_filtered = cur_X_filtered.values

            cur_y = datas[df_name]['y_train']

            regression = df_name == 'bike'

            cur_X_results = cross_validate(
                X=cur_X_filtered, y=cur_y.values.reshape(-1), progress=False,
                model=model, regression=regression, n_splits=3
            )
            results[df_name][filter_name] = cur_X_results
            i += 1
    return results


def filter_cluster_learner_check(final_elbows, datas, model_grid, y_scaled=False):
    results = {}
    i = 1
    for df_name in final_elbows:
        results[df_name] = {}
        for filter_name in final_elbows[df_name]:
            results[df_name][filter_name] = {}
            for cluster_type in final_elbows[df_name][filter_name]:
                print(f'\r{i}/16. Df: {df_name}. Filter: {filter_name}, clustering: {cluster_type}.', end='')
                regression = df_name == 'bike'

                if regression:
                    model_class, model_kwargs = model_grid['regression']
                else:
                    model_class, model_kwargs = model_grid['classification']

                model = model_class(**model_kwargs)
                cur_clusters = final_elbows[df_name][filter_name][cluster_type]['preds']
                cur_X_filtered = datas[df_name][f'X_train_{filter_name}']

                if type(cur_X_filtered) == pd.DataFrame:
                    cur_X_filtered = cur_X_filtered.values

                cur_X = np.hstack([cur_X_filtered, cur_clusters.reshape(-1, 1)])
                if y_scaled:
                    cur_y = datas[df_name]['y_train_scaled']
                else:
                    cur_y = datas[df_name]['y_train']

                regression = df_name == 'bike'

                cur_X_results = cross_validate(
                    X=cur_X, y=cur_y.values.reshape(-1), progress=False,
                    model=model, regression=regression, n_splits=3
                )
                results[df_name][filter_name][cluster_type] = cur_X_results
                i += 1
    return results


def plot_fcl_results(fcl_results, fl_results, original_data_results):
    summary = []
    for df_name in fcl_results:
        cur_metric = 'mse' if df_name == 'bike' else 'accuracy'
        for filter_name in fcl_results[df_name]:
            cur_fl_results = fl_results[df_name][filter_name]
            summary.append({
                'combination': f'{filter_name}',
                'mean_duration': cur_fl_results['mean_duration'],
                'mean_performance': cur_fl_results[f'mean_{cur_metric}'],
                'dataset': df_name
            })
            for cluster_type in fcl_results[df_name][filter_name]:
                cur_results = fcl_results[df_name][filter_name][cluster_type]
                summary.append({
                    'combination': f'{filter_name}+{cluster_type}',
                    'mean_duration': cur_results['mean_duration'],
                    'mean_performance': cur_results[f'mean_{cur_metric}'],
                    'dataset': df_name
                })

    summary = pd.DataFrame(summary)
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches((18, 16))

    for num, df_name in enumerate(summary['dataset'].unique()):
        dataset_metric = 'mse' if df_name == 'bike' else 'accuracy'
        color = 'tab:blue' if df_name == 'bike' else 'tab:orange'

        cur_summary = (
            summary
            .query(f'dataset == "{df_name}"')
            .set_index('combination')
            .sort_values('mean_performance', ascending=(df_name == 'wine'))
        )

        # Performance limis
        perf_min_bound = cur_summary['mean_performance'].min()
        perf_max_bound = cur_summary['mean_performance'].max()

        perf_offset = perf_max_bound * 0.1
        perf_min_bound, perf_max_bound = (perf_min_bound - perf_offset), (perf_max_bound + perf_offset)

        # Duration limis
        dur_min_bound = cur_summary['mean_duration'].min()
        dur_max_bound = cur_summary['mean_duration'].max()

        dur_offset = dur_max_bound * 0.1
        dur_min_bound, dur_max_bound = (dur_min_bound - dur_offset), (dur_max_bound + dur_offset)

        cur_summary['mean_performance'].plot(
            kind='barh', ax=axes[0][num], title=f'{df_name.capitalize()} performance', color=color,
            label='transformed data'
        )
        cur_summary['mean_duration'].plot(
            kind='barh', ax=axes[1][num], title=f'{df_name.capitalize()} time duration', color=color,
            label='transformed data'
        )

        axes[0][num].set_xlabel(f'mean cv {dataset_metric}')
        axes[1][num].set_xlabel('seconds')

        axes[0][num].axvline(
            original_data_results[df_name][f'mean_{dataset_metric}'], linestyle='--', alpha=0.5,
            label='original data'
        )

        axes[1][num].axvline(
            original_data_results[df_name][f'mean_duration'], linestyle='--', alpha=0.5,
            label='original data'
        )

        axes[0][num].set_xlim((perf_min_bound, perf_max_bound))
        axes[1][num].set_xlim((dur_min_bound, dur_max_bound))

        axes[0].legend()
        axes[1][num].legend()

    for ax in axes.ravel():
        ax.tick_params(axis='y', which='major', labelsize=15)

    return fig, axes
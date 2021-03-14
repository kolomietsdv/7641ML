import networkx.generators.random_graphs as rg
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import time
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sps

plt.style.use('ggplot')


def time_it(f, *args, **kwargs):
    time_start = time.time()
    res = f(*args, **kwargs)
    duration = time.time() - time_start
    return res, duration


def run_maxk_colors(n, p=0.35, candidates=False, base_kwargs=False, k_max=15):
    g = rg.erdos_renyi_graph(n, p)
    fitness = mlrose.MaxKColor(list(g.edges))
    # problem = mlrose.DiscreteOpt(length=n, fitness_fn=fitness, maximize=False, max_val=k)

    if not candidates:
        candidates = {
            'mimic': mlrose.mimic, 'sa': mlrose.simulated_annealing,
            'rhc': mlrose.random_hill_climb, 'ga': mlrose.genetic_alg
        }
    if not base_kwargs:
        base_kwargs = {
            'mimic': dict(curve=True), 'sa': dict(curve=True),
            'rhc': dict(curve=True, restarts=100), 'ga': dict(curve=True)
        }

    final_results = {}

    for name, alg in candidates.items():
        for k in range(2, k_max+1):
            problem = mlrose.DiscreteOpt(length=n, fitness_fn=fitness, maximize=False, max_val=k)
            final_results.setdefault(name, {})
            (best_state, best_fitness, curve), duration = time_it(alg, problem, **base_kwargs[name])
            final_results[name][k] = {
                'best_state': best_state,
                'best_fitness': best_fitness,
                'curve': curve,
                'duration': duration
            }
            if best_fitness == 0:
                break
    return final_results


def search_par(problem, alg, par_name, range, alg_pars, time_bonus=0):
    results = {}
    for par_val in tqdm(range):

        alg_pars[par_name] = par_val
        # pars.update({'curve': True, 'max_iters': 1e5})

        (best_state, best_fitness, curve), duration = time_it(alg, problem, **alg_pars)
        time_bonus += time_bonus
        results[par_val] = {
            'best_state': best_state,
            'best_fitness': best_fitness,
            'curve': curve,
            'duration': duration
        }
    return results


def massive_search_par(problem, candidates, par_spaces, alg_pars, time_bonus=0):
    results = {}
    for name, alg in tqdm(candidates.items()):
        results[name] = search_par(problem, alg, alg_pars=alg_pars[name],
                                   par_name=par_spaces[name]['par_name'],
                                   range=par_spaces[name]['range'], time_bonus=time_bonus)
    return results


# def plot_massive_search_results():
#     best_fitness



def plot_maxk_colors(results):
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(20, 12)

    N = sorted([x for x in results])
    algs = [x for x in list(results.values())[0]]

    # Best fitness
    for alg in algs:
        axes[0][0].plot(N, [results[n][alg]['best_k'] for n in N], label=alg)
    axes[0][0].set_title('Best found value of K')
    axes[0][0].legend()

    # Best fitness
    for alg in algs:
        axes[0][1].plot(N, [results[n][alg]['best_fitness'] for n in N], label=alg)
    axes[0][1].set_title('Edges with the same colors')
    axes[0][1].legend()

    # Time duration
    for alg in algs:
        axes[1][0].plot(N, [results[n][alg]['duration'] for n in N], label=alg)
    axes[1][0].set_title('Time duration')
    axes[1][0].legend()

    # Evaluations
    for alg in algs:
        axes[1][1].plot(N, [results[n][alg]['evals'] for n in N], label=alg)
    axes[1][1].set_title('Number of function evaluations')
    axes[1][1].legend()

    return fig, axes

def preag_kcolor_results(res):
    agg_res = {}
    for n in res:
        agg_res.setdefault(n, {})
        for alg_name in res[n]:
            best_k_results = max([x for x in res[n][alg_name].items()], key=lambda x: x[0])
            agg_res[n][alg_name] = {
                'best_k': best_k_results[0],
                'best_fitness': best_k_results[1]['best_fitness'],
                'duration': best_k_results[1]['duration'],
                'evals': len(best_k_results[1]['curve'])
            }
    return agg_res


def run_four_peaks(n, t_pct=0.1, attempts=10, six=False, candidates=False, base_kwargs=False):
    if six:
        fitness = mlrose.SixPeaks(t_pct=t_pct)
    else:
        fitness = mlrose.FourPeaks(t_pct=t_pct)
    problem = mlrose.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=2)

    if not candidates:
        candidates = {
            'mimic': mlrose.mimic, 'sa': mlrose.simulated_annealing,
            'rhc': mlrose.random_hill_climb, 'ga': mlrose.genetic_alg
        }
    if not base_kwargs:
        base_kwargs = dict(curve=True)

    final_results = {}
    for at in range(attempts):
        for name, alg in candidates.items():
            (best_state, best_fitness, curve), duration = time_it(alg, problem, **base_kwargs)
            final_results.setdefault(name, {})
            final_results[name][at] = {
                'best_state': best_state,
                'best_fitness': best_fitness,
                'curve': curve,
                'duration': duration
            }
    return final_results


def run_problem(problem, problem_kwargs, length, attempts=10, candidates=False, base_kwargs=False):

    fitness = problem(**problem_kwargs)
    problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)

    if not candidates:
        candidates = {
            'mimic': mlrose.mimic, 'sa': mlrose.simulated_annealing,
            'rhc': mlrose.random_hill_climb, 'ga': mlrose.genetic_alg
        }
    if not base_kwargs:
        base_kwargs = {x: dict(curve=True) for x in candidates}

    final_results = {}
    for at in range(attempts):
        for name, alg in candidates.items():
            (best_state, best_fitness, curve), duration = time_it(alg, problem, **base_kwargs[name])
            final_results.setdefault(name, {})
            final_results[name][at] = {
                'best_state': best_state,
                'best_fitness': best_fitness,
                'curve': curve,
                'duration': duration
            }
    return final_results


def plot_problem_results(results):
    fig, axes = plt.subplots(3, 1)
    fig.set_size_inches(8, 15)

    N = [x for x in results]
    algs = [x for x in list(results.values())[0]]


    # Best fitness
    for alg in algs:
        axes[0].plot(N, [results[n][alg]['best_fitness'] for n in N], label=alg)
    axes[0].set_title('Best found value')
    axes[0].legend()

    # Time duration
    for alg in algs:
        axes[1].plot(N, [results[n][alg]['duration'] for n in N], label=alg)
    axes[1].set_title('Time duration')
    axes[1].legend()

    # Evaluations
    for alg in algs:
        axes[2].plot(N, [len(results[n][alg]['curve']) for n in N], label=alg)
    axes[2].set_title('Number of function evaluations')
    axes[2].legend()

    return fig, axes


def preprocess_n(results):
    new_results = {}
    for n in results:
        n_results = results[n]
        new_results.setdefault(n, {})
        for alg_name in n_results:
            alg_results = n_results[alg_name]
            alg_results_pd = pd.DataFrame(alg_results).T

            # best fitness
            bf_ser = alg_results_pd['best_fitness']
            bf_mean, bf_std, bf_max = bf_ser.mean(), bf_ser.std(), bf_ser.max()

            # duration
            dur_ser = alg_results_pd['duration']
            dur_mean, dur_std, dur_max = dur_ser.mean(), dur_ser.std(), dur_ser.min()

            # evals
            evals = alg_results_pd['curve'].apply(len).unique()[0]

            new_results[n][alg_name] = {
                'best_fitness': (bf_mean, bf_std, bf_max),
                'duration': (dur_mean, dur_std, dur_max),
                'evals': evals
            }
    return new_results



def plot_four_peaks_results(results):
    fig, axes = plt.subplots(3, 1)
    fig.set_size_inches(8, 15)

    N = [x for x in results]
    algs = [x for x in list(results.values())[0]]

    # Best fitness
    for alg in algs:
        # mean_alg_values = np.array([results[n][alg]['best_fitness'][0] for n in N])
        # std_alg_values = np.array([results[n][alg]['best_fitness'][1] for n in N])
        # axes[0].plot(N, mean_alg_values, label=alg)
        # axes[0].fill_between(N, (mean_alg_values - std_alg_values), (mean_alg_values + std_alg_values),
        #                      alpha=.1, label=alg)
        axes[0].plot(N, [results[n][alg]['best_fitness'][2] for n in N], label=alg)
    axes[0].set_title('Best found value')
    axes[0].legend()

    # Time duration
    for alg in algs:
        # mean_alg_values = np.array([results[n][alg]['duration'][0] for n in N])
        # std_alg_values = np.array([results[n][alg]['duration'][1] for n in N])
        # axes[1].plot(N, mean_alg_values, label=alg)
        # axes[1].fill_between(N, (mean_alg_values - std_alg_values), (mean_alg_values + std_alg_values),
        #                      alpha=.1, label=alg)
        axes[1].plot(N, [results[n][alg]['duration'][2] for n in N], label=alg)
    axes[1].set_title('Time duration')
    axes[1].legend()

    # Evaluations
    for alg in algs:
        axes[2].plot(N, [results[n][alg]['evals'] for n in N], label=alg)
    axes[2].set_title('Number of function evaluations')
    axes[2].legend()

    return fig, axes


def single_problem_plot(massive_sr):
    massive_sr_pds = {}
    par_names = {'ga': 'pop_size', 'mimic': 'pop_size', 'rhc': 'restarts'}

    for name, df in massive_sr.items():
        if name == 'sa':
            sa_r = pd.DataFrame(massive_sr[name]).T.reset_index()
            sa_r['decay'] = sa_r['index'].apply(lambda x: x.decay)
            sa_r.set_index('decay', inplace=True)
            massive_sr_pds[name] = sa_r
        else:
            massive_sr_pds[name] = pd.DataFrame(massive_sr[name]).T.sort_index()
            massive_sr_pds[name].index.name = par_names[name]
        massive_sr_pds[name]['evals'] = massive_sr_pds[name]['curve'].apply(lambda x: len(x))


    fig, axes = plt.subplots(3, len(massive_sr_pds))
    fig.set_size_inches(5*len(massive_sr_pds), 5 * 3)
    num = 0

    mets = ['best_fitness', 'duration', 'evals']
    colors = dict(zip(mets, ['tab:red', 'tab:blue', 'tab:green']))

    for met in ['best_fitness', 'duration', 'evals']:
        for name, df in massive_sr_pds.items():

            max_fitness = max(x[met].max() for x in massive_sr_pds.values())
            min_fitness = min(x[met].min() for x in massive_sr_pds.values())

            cur_ax = axes[num // len(massive_sr_pds)][num % len(massive_sr_pds)]

            if met == 'evals':
                logy = True
            else:
                logy = False

            df[met].plot(ax=cur_ax, logy=logy, color=colors[met])

            if num % len(massive_sr_pds) == 0:
                cur_ax.set_ylabel(met)
            else:
                cur_ax.set_yticklabels('')

            if num // len(massive_sr_pds) == 0:
                cur_ax.set_title(name)

            val = abs(max_fitness * 0.1)
            cur_ax.set_ylim((min_fitness - val, max_fitness + val))

            cur_ax.axhline(min_fitness, linestyle='--', alpha=0.5, label='min')
            cur_ax.axhline(max_fitness, linestyle='--', alpha=0.5, label='max')
            num += 1
    plt.tight_layout()
    return fig, axes, massive_sr_pds

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
    #             print(encoded[-1].shape)
    #     X = np.hstack(encoded)
    #         print(encoded[-1].shape)
    if is_col_object.sum() > 0:
        X = sps.hstack(encoded)
    else:
        X = np.hstack(encoded)
    return X, y, trained_ohes
    # for par in ['']


def build_lc(estimator, pars, train, train_y, valid, valid_y, metric_f, metric_f_name,
             percent_range=range(10, 103, 3)):
    np.random.seed(13)
    metrics_dict = {}

    estmr = estimator(**pars)

    for i in percent_range:
        random_index = np.random.choice(train.shape[0],
                                        replace=False,
                                        size=min(train.shape[0], round(train.shape[0] * i / 100)))

        _, duration = time_it(estmr.fit, train[random_index], train_y[random_index])

        train_preds = estmr.predict(train[random_index])
        preds = estmr.predict(valid)

        metrics = {}

        metrics[f'test_{metric_f_name}'] = metric_f(valid_y, preds)
        metrics[f'train_{metric_f_name}'] = metric_f(train_y[random_index], train_preds)

        metrics['fit_time'] = duration

        metrics_dict[i] = metrics
        print(f"\r{i}% Ammount: {random_index.shape[0]} {duration:.4f} seconds, train: {metrics[f'train_{metric_f_name}']:.3f}, test: {metrics[f'test_{metric_f_name}']:.3f}", end='')

    perc = sorted(metrics_dict.keys())
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(9, 4)

    axes[0].plot(perc,
                 [metrics_dict[x][f'test_{metric_f_name}'] for x in perc], color='red',
                 label='test')
    axes[0].plot(perc,
                 [metrics_dict[x][f'train_{metric_f_name}'] for x in perc], color='orange',
                 label='train')

    axes[0].legend()
    axes[0].set_title(f'{metric_f_name}')
    axes[0].set_xlabel('Ammount of the training set, percent')
    # axes[0].set_ylim(0.5, 1)

    axes[1].plot(perc,
                 [metrics_dict[x]['fit_time'] for x in perc], color='blue')
    axes[1].set_title('fit_time')
    axes[1].set_xlabel('Ammount of the training set, percent')

    fig.suptitle('Train volume effect', fontsize=15)

    plt.tight_layout()
    return fig, axes


def optimization_test(par_spaces, X, y, valid_X, valid_y, target_metrics):
    res = {}
    for alg_name, par_space in tqdm(par_spaces.items()):
        res.setdefault(alg_name, {})
        res[alg_name]['par_name'] = par_space['par_name']
        res[alg_name].setdefault('range', {})
        print(alg_name)
        for par_val in tqdm(par_space['range']):
            res[alg_name]['range'].setdefault(par_val, {})
            nn_model = mlrose.NeuralNetwork(
                hidden_nodes=[15 for _ in range(9)],
                activation='relu', algorithm=alg_name,
                max_iters=2000, bias=True, is_classifier=False, learning_rate=0.1,
                early_stopping=False, clip_max=0.5, max_attempts=100,
                random_state=13, curve=True, **{par_space['par_name']: par_val}
            )
            _, duration = time_it(nn_model.fit, **{'X': X, 'y': y})
            preds = nn_model.predict(valid_X)
            res[alg_name]['range'][par_val]['preds'] = preds
            res[alg_name]['range'][par_val]['fitness_curve'] = nn_model.fitness_curve
            res[alg_name]['range'][par_val]['weights'] = nn_model.fitted_weights
            res[alg_name]['range'][par_val]['duration'] = duration
            for m_name, m_f in target_metrics.items():
                res[alg_name]['range'][par_val][m_name] = m_f(y_true=valid_y, y_pred=preds)
    return res


def plot_wo_results(wo_res):
    wo_pds = {}
    # par_names = {'ga': 'pop_size', 'mimic': 'pop_size', 'rhc': 'restarts'}
    for alg_name in wo_res:
        if alg_name == 'simulated_annealing':
            sa_r = pd.DataFrame(wo_res[alg_name]['range']).T.reset_index()
            sa_r['decay'] = sa_r['index'].apply(lambda x: x.decay)
            sa_r.set_index('decay', inplace=True)
            wo_pds[alg_name] = sa_r
        else:
            wo_pds[alg_name] = pd.DataFrame(wo_res[alg_name]['range']).T.sort_index()
            wo_pds[alg_name].index.name = wo_res[alg_name]['par_name']
        wo_pds[alg_name]['evals'] = wo_pds[alg_name]['fitness_curve'].apply(lambda x: len(x))

    fig, axes = plt.subplots(3, len(wo_pds))
    fig.set_size_inches(5 * len(wo_pds), 12)
    num = 0

    mets = ['MSE', 'duration', 'evals']
    colors = dict(zip(mets, ['tab:red', 'tab:blue', 'tab:green']))

    for met in ['MSE', 'duration', 'evals']:
        for name, df in wo_pds.items():

            max_fitness = max(x[met].max() for x in wo_pds.values())
            min_fitness = min(x[met].min() for x in wo_pds.values())

            cur_ax = axes[num // len(wo_pds)][num % len(wo_pds)]

            if met == 'evals':
                logy = True
            else:
                logy = False

            df[met].plot(ax=cur_ax, color=colors[met])

            if num % len(wo_pds) == 0:
                cur_ax.set_ylabel(met)
            else:
                cur_ax.set_yticklabels('')

            if num // len(wo_pds) == 0:
                cur_ax.set_title(name)

            val = abs(max_fitness*0.1)
            cur_ax.set_ylim((min_fitness - val, max_fitness + val))
            cur_ax.axhline(min_fitness, linestyle='--', alpha=0.5, label='min')
            cur_ax.axhline(max_fitness, linestyle='--', alpha=0.5, label='max')
            num += 1

    plt.tight_layout()
    return fig, axes, wo_pds


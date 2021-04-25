import time
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

class VI:
    def __init__(self, gamma=0.1, theta=1e-5):
        self.V = {}
        self.P = {}
        self.gamma = gamma
        self.theta = theta
        self.deltas = []
        self.iter_stats = []
        self.sweeps = 0

    def iterate(self, env):
        delta = 0
        delta_sum = 0
        policy_changes = 0

        self.sweeps += 1

        for s in env.P.keys():
            # print(s)
            self.V.setdefault(s, 0)
            cur_v = self.V[s]

            # Seeking the best action
            action_vs = []
            for a in env.P[s].keys():
                action_v = 0
                # print(env.P[s][a])
                for p, next_s, r, end in env.P[s][a]:
                    self.V.setdefault(next_s, 0)
                    action_v += p * (r + self.gamma * self.V[next_s])

                action_vs.append((a, action_v))
            # print(action_vs)
            best_a, best_a_value = max(action_vs, key=lambda x: (x[1], x[0]))

            if self.P.get(s, 'none') != best_a:
                policy_changes += 1

            # Assign the best action to the state
            self.V[s] = best_a_value
            self.P[s] = best_a

            # Check delta for convergence
            delta = max(delta, abs(self.V[s] - cur_v))

            # Sum statistics
            delta_sum += abs(self.V[s] - cur_v)

        return delta, {'delta_sum': delta_sum, 'policy_changes': policy_changes}

    def solve(self, env, iters=10, check_trials=100):
        self.V = {}
        self.P = {}
        self.deltas = []
        self.iter_stats = []
        self.sweeps = 0

        for i in range(iters):
            delta, stats = self.iterate(env)

            stats['result_check'] = multi_check(env, self.P, times=check_trials, max_steps=50)
            self.deltas.append(delta)
            self.iter_stats.append(stats)

            if i%5 == 0:
                print(f'\rIteration: {i}/{iters}. Delta: {delta:.3f}', end='')

            if delta < self.theta:
                break

        print(f'\rDone in {i} iterations.' + ' ' * 20)
        return self.P


def get_available_actions(env, s):
    return list(env.P[s].keys())


def init_new_s(env, s, default_value=0):
    return {x: default_value for x in get_available_actions(env, s)}


class PI:
    def __init__(self, gamma=0.1, theta=1e-5):
        self.V = {}
        self.P = {}
        self.gamma = gamma
        self.theta = theta
        self.deltas = []
        self.iter_stats = []
        self.sweeps = 0

    def evaluate_policy(self, env, max_iter=1000):
        delta_sum = 0

        for i in range(max_iter):
            delta = 0
            self.sweeps += 1
            # Policy evaluation
            for s in env.P.keys():

                self.V.setdefault(s, 0)
                cur_v = self.V[s]

                # Initialize random policy for the state if there is no policy here
                self.P.setdefault(s, np.random.choice(get_available_actions(env, s)))
                a = self.P[s]

                new_val = 0
                for p, next_s, r, end in env.P[s][a]:
                    self.V.setdefault(next_s, 0)
                    new_val += p * (r + self.gamma * self.V[next_s])

                self.V[s] = new_val

                # Check delta for convergence
                delta = max(delta, abs(self.V[s] - cur_v))

                delta_sum += abs(self.V[s] - cur_v)

            if delta < self.theta:
                return True, delta_sum
        return False, delta_sum


    def greedify_state(self, env, s):
        action_vs = []
        for a in env.P[s].keys():
            action_v = 0
            for p, next_s, r, end in env.P[s][a]:
                self.V.setdefault(next_s, 0)
                action_v += p * (r + self.gamma * self.V[next_s])

            action_vs.append((a, action_v))

        # Searching highest value
        best_a, best_value = max(action_vs, key=lambda x: (x[1], x[0]))

        # # Shatter ties on actions
        # best_a = np.random.choice([x[0] for x in action_vs if x[1] == best_value])
        return best_a

    def improve_policy(self, env):
        policy_changes = 0
        policy_stable = True
        self.sweeps += 1
        for s in self.P.keys():
            old_a = self.P[s]
            self.P[s] = self.greedify_state(env, s)
            if self.P[s] != old_a:
                policy_changes += 1
                policy_stable = False

        return policy_stable, policy_changes

    def solve(self, env, iters=10, check_trials=100):
        self.V = {}
        self.P = {}
        self.deltas = []
        self.iter_stats = []
        self.sweeps = 0

        for i in range(1, iters+1):
            _, delta_sum = self.evaluate_policy(env)
            policy_stable, policy_changes = self.improve_policy(env)

            # if i % 5 == 0:
            print(f'\rIteration: {i}/{iters}.', end='')

            self.iter_stats.append({
                 'delta_sum': delta_sum,
                 'policy_changes': policy_changes,
                 'result_check': multi_check(env, self.P, times=check_trials, max_steps=50)
            })
            if policy_stable:
                break



        print(f'\r Policy has {"" if policy_stable else "not "}converged in {i} iterations.' + ' ' * 20)
        return self.P


class Dyna:
    def __init__(self, alpha=1e-2, epsilon=0.1, gamma=0.5, choice_strategy='eg', planning_steps=10,
                 alpha_decay=1):
        self.Q = {}
        self.M = {}
        self.planning_steps = planning_steps
        self.alpha_decay = alpha_decay
        self.counts = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.C = {}
        self.C_out = {}
        self.P = {}
        self.choice_strategy = choice_strategy
        self.stats = {}

    def run_episode(self, env, render=False, pick_best=False, steps_lim=1e10, true_v=None):
        R = 0
        error_sum = 0
        p_error = 0
        steps = 0
        done = False
        s = env.reset()
        poss_actions = get_available_actions(env, s)
        self.Q.setdefault(s, {a: 0 for a in poss_actions})
        # self.M.setdefault(s, {})

        a = self.choose_action(env, s, pick_best=pick_best)

        # and (steps < 500)
        while (not done) and (steps < steps_lim):
            if render:
                env.render()
                time.sleep(1)
                print(f'cur_s: {s}, chosen_action: {a}')

            steps += 1
            new_s, r, done, info = env.step(a)
            if render:
                print(f'new_s: {new_s}, reward: {r}')
                print('--' * 50)

            poss_actions = get_available_actions(env, new_s)
            self.Q.setdefault(new_s, {a: 0 for a in poss_actions})

            new_action = self.choose_action(env=env, s=new_s, pick_best=pick_best)
            # new_action = self.epsilon_greedy(env, new_s, eps)

            # td_target = r + self.gamma * self.Q[new_s][new_action]
            if not pick_best:
                td_target = r + self.gamma * max(
                    self.Q.get(new_s, init_new_s(env, new_s)).items(), key=lambda x: (x[1], x[0])
                )[1]
                td_error = td_target - self.Q[s][a]
                self.Q[s][a] += self.alpha * td_error

                self.replenish(s, a, r, new_s)
                planning_errors = self.plan(self.planning_steps)

                error_sum += abs(self.alpha * td_error)
                p_error += abs(self.alpha * planning_errors)

            s, a = new_s, new_action
            R += r

        if true_v:
            true_v_error = 0
            for s in true_v:
                if s in self.Q:
                    estimated_value = max(self.Q[s].values())
                else:
                    estimated_value = 0
                true_v_error += abs(true_v[s] - estimated_value)

            return R, error_sum, p_error, true_v_error

        return R, error_sum, p_error

    def replenish(self, s, a, r, new_s):
        self.M.setdefault(s, {})
        self.M[s].setdefault(a, {})
        self.M[s][a].setdefault(new_s, r)

        # To estimate probability of ending in new_s after a
        self.C_out.setdefault(s, {})
        self.C_out[s].setdefault(a, {})
        self.C_out[s][a][new_s] = self.C_out[s][a].get(new_s, 0) + 1

    def choose_action(self, env, s, pick_best=False):
        if pick_best:
            # print(f'picking the best for {s}')
            # print(self.Q[s])
            new_action = self.pick_best(s=s)
            # print(f'chosen_action: {new_action}')
        elif self.choice_strategy == 'eg':
            new_action = self.epsilon_greedy(env=env, s=s)
        elif self.choice_strategy == 'ucb':
            new_action = self.UCB(env=env, s=s)
        elif self.choice_strategy == 'pb':
            new_action = self.pick_best(s=s)
        else:
            raise ValueError
        return new_action

    def plan(self, steps=10):
        errors = 0
        for i in range(min(len(self.M), steps)):
            random_idx = np.random.choice(len(self.M.keys()))
            s = list(self.M.keys())[random_idx]
            a = np.random.choice(list(self.M[s].keys()))

            td_target = 0
            total_count = sum(self.C_out[s][a].values())

            for new_s in self.M[s][a]:
                r = self.M[s][a][new_s]
                prob = self.C_out[s][a][new_s]/total_count
                td_target += prob * (r + self.gamma * max(self.Q.get(new_s, {0: 0}).items(),
                                                          key=lambda x: (x[1], x[0]))[1])

            td_error = td_target - self.Q[s][a]
            self.Q[s][a] += self.alpha * td_error
            errors += abs(td_error)

        return errors

    def learn(self, env, episodes=1000, counter_lim=10, delta_lim=1e-5, true_v=None):
        counter = 0
        self.Q = {}
        self.C = {}
        self.M = {}

        self.stats = {'Returns': [], 'Error': [], 'Planning Error': [], 'Policy amendments': [],
                      'True V error': []}

        for i in range(episodes):
            if true_v:
                R, error_sum, p_error_sum, true_v_error = self.run_episode(env, true_v=true_v)
                self.stats['True V error'].append(true_v_error)
            else:
                R, error_sum, p_error_sum  = self.run_episode(env)

            policy_amendments = self.build_policy()
            # self.stats['Returns'].append(R)
            self.stats['Error'].append(error_sum)
            self.stats['Planning Error'].append(p_error_sum)
            self.stats['Policy amendments'].append(policy_amendments)
            self.stats['Returns'].append(self.run_episode(env, pick_best=True, steps_lim=100)[0])
            self.alpha *= self.alpha_decay

            print(f'\r{i}/{episodes}, alpha={self.alpha:.3f}, error: {error_sum:.2f}', end='')

            # if policy_amendments == 0:
            #     counter += 1
            #     if counter > counter_lim:
            #         break
            # else:
            #     counter = 0
            # if (error_sum + p_error_sum) < delta_lim and i > 10:
            #     break

        return self.stats

    def epsilon_greedy(self, **kwargs):
        s, env = kwargs['s'], kwargs['env']

        A = get_available_actions(env, s)
        for a in A:
            self.Q[s].setdefault(a, 0)

        if np.random.random() <= self.epsilon:
            return np.random.choice(A)
        else:
            best_a, highest_q = max(self.Q[s].items(), key=lambda x: x[1])
            return np.random.choice([x[0] for x in self.Q[s].items() if x[1] == highest_q])

    def UCB(self, **kwargs):
        s, env = kwargs['s'], kwargs['env']

        # t
        self.C.setdefault(s, {'state': 0})
        self.C[s]['state'] += 1

        A = get_available_actions(env, s)

        action_values = []
        for a in A:
            self.Q[s].setdefault(a, 0)
            if a not in self.C[s]:
                self.C[s][a] = 1
                action_values.append((a, 999))
            else:
                action_values.append((
                    a, self.Q[s][a] + np.sqrt(np.log(self.C[s]['state']) / self.C[s][a])
                ))

        best_a, _ = max(action_values, key=lambda x: (x[1], x[0]))
        self.C[s][best_a] += 1
        return best_a

    def pick_best(self, **kwargs):
        _, best_action_value = max(self.Q[kwargs['s']].items(), key=lambda x: (x[1], x[0]))
        chosen_a = np.random.choice([a[0] for a in self.Q[kwargs['s']].items() if a[1] == best_action_value])
        return chosen_a

    def build_policy(self, env=None):
        policy_amendments = 0
        states = env.S if env else self.Q
        for s in states:
            if env:
                avail = get_available_actions(env, s)
                self.P.setdefault(s, np.random.choice(avail))
            else:
                self.P.setdefault(s, 0)

            old_p = self.P[s]
            if s in self.Q:
                self.P[s] = max(self.Q[s].items(), key=lambda x: (x[1], x[0]))[0]

            if self.P[s] != old_p:
                policy_amendments += 1

        return policy_amendments


def check_policy(env, P, render=False, max_steps=1000, verbose=0):
    done = False
    R, steps = 0, 0

    s = env.reset()
    while (not done) and (steps <= max_steps):
        a = P[s]
        s, r, done, info = env.step(a)

        steps += 1
        R += r

        if render:
            print(list(env.decode(env.s)))
            env.render()
            time.sleep(1)
        elif verbose > 0:
            print(f'\rStep: {steps}.', end='')
    return R


def time_it(f, *args, **kwargs):
    time_start = time.time()
    res = f(*args, **kwargs)
    duration = time.time() - time_start
    return res, duration


def multi_check(env, P, times=10, max_steps=1000):
    results = {'trials': []}

    for t in range(times):
        results['trials'].append(check_policy(env, P, False, max_steps=max_steps))
    results['trials'] = np.array(results['trials'])

    for f_name, f in {
        'mean': np.mean, 'median': np.median, 'min': np.min, 'max': np.max, 'std': np.std, 'sum': np.sum
    }.items():
        results[f_name] = f(results['trials']).item()

    return results


def visualize_taxi_policy(P, env):
    taxi_map = [
        '+---------+',
        '| : | : : |',
        '| : | : : |',
        '| : : : : |',
        '| | : | : |',
        '| | : | : |',
        '+---------+'
    ]

    arrow_dict = dict(zip([1, 0, 3, 2, 4, 5, 6], ['\u2191', '\u2193', '\u2190', '\u2192', 'p', 'd', '?']))

    print("Before picking up:")
    pick_up = taxi_map[:]
    for i in range(5):
        for j in range(5):
            s = env.encode(*[i, j, 0, 1])

            new_row = list(pick_up[i + 1])
            new_row[(j * 2 + 1)] = arrow_dict[P[s]]

            pick_up[i + 1] = ''.join(new_row)
    print(*pick_up, sep='\n')

    print("After picking up:")
    drop_off = taxi_map[:]
    for i in range(5):
        for j in range(5):
            s = env.encode(*[i, j, 4, 1])
            # if (i, j) == (4, 1): print(s, P.get(s, 6))
            new_row = list(drop_off[i + 1])
            new_row[(j * 2 + 1)] = arrow_dict[P.get(s, 6)]

            drop_off[i + 1] = ''.join(new_row)
    print(*drop_off, sep='\n')


def visualize_taxi_values(V, env, Q=False):

    # arrow_dict = dict(zip([1, 0, 3, 2, 4, 5], ['\u2191', '\u2193', '\u2190', '\u2192', 'p', 'd']))

    print("Before picking up:")
    pick_up = np.zeros(shape=(5, 5))
    for i in range(5):
        for j in range(5):
            s = env.encode(*[i, j, 0, 1])
            if not Q:
                pick_up[i, j] = str(round(V[s], 2))
            else:
                pick_up[i, j] = str(round(max(V.get(s, {0: 0}).items(), key=lambda x: x[1])[1], 3))
    print(pick_up)

    print("After picking up:")
    drop_off = np.zeros(shape=(5, 5))
    for i in range(5):
        for j in range(5):
            s = env.encode(*[i, j, 4, 1])
            if not Q:
                drop_off[i, j] = str(round(V[s], 2))
            else:
                drop_off[i, j] = str(round(max(V.get(s, {0: 0}).items(), key=lambda x: x[1])[1], 3))
    print(drop_off)


def extract_results_check(solutions, gamma, stat):
    return list(map(lambda x: x['result_check'][stat], solutions[gamma].iter_stats))


def extract_iter_stats(solutions, gamma, stat):
    return list(map(lambda x: x[stat], solutions[gamma].iter_stats))


def _fill_row(vi_solutions, axes, row, gammas, colors, logy_col1=False, logy_col2=False, fb=True):
    for gamma in gammas:
        pd.Series(extract_iter_stats(vi_solutions, gamma=gamma, stat='delta_sum')).plot(
            ax=axes[row][0], color=colors[gamma], label=gamma, logy=logy_col1
        )
        axes[row][0].set_title('Delta sums.', fontsize=20)
        pd.Series(extract_iter_stats(vi_solutions, gamma=gamma, stat='policy_changes')).plot(
            ax=axes[row][1], color=colors[gamma], label=gamma, logy=logy_col2
        )
        axes[row][1].set_title('Policy amendments.', fontsize=20)
        means = np.array(extract_results_check(vi_solutions, gamma=gamma, stat='mean'))
        # stds = np.array(extract_results_check(vi_solutions, gamma=gamma, stat='std'))
        mins = np.array(extract_results_check(vi_solutions, gamma=gamma, stat='min'))
        maxes = np.array(extract_results_check(vi_solutions, gamma=gamma, stat='max'))

        pd.Series(means).plot(ax=axes[row][2], color=colors[gamma], label=gamma)
        if fb:
            axes[row][2].fill_between(range(0, len(mins)), mins, maxes, alpha=0.3, color=colors[gamma])

        axes[row][2].set_title('Returns.', fontsize=20)
        [axes[row][x].legend() for x in range(3)]


def plot_vi_results(vi_solutions, logy_col1=False, logy_col2=False, fb=True, one_row=False):
    gammas = [x for x in vi_solutions.keys()]

    fig, axes = plt.subplots(2, 3, sharex=True)
    fig.set_size_inches(20, 10)

    colors = dict(zip(gammas, [f'tab:{x}' for x in ['blue', 'green', 'red', 'orange']]))

    _fill_row(vi_solutions, axes, 0, gammas[:2], colors, logy_col1=logy_col1, logy_col2=logy_col2, fb=fb)
    _fill_row(vi_solutions, axes, 0 if one_row else 1, gammas[2:], colors, logy_col1=logy_col1, logy_col2=logy_col2, fb=fb)

    for ax in axes.flatten():
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)

    return fig, axes


def plot_dyna_results(results, each=10):
    rows = int(np.ceil(len(results)/2))
    fig, axes = plt.subplots(rows, 2)
    fig.set_size_inches(15, 5*rows)

    for num, key in enumerate(results):
        logy = False if num == 0 else True
        cur_ax = axes[num // 2][num % 2]

        plot_data = pd.Series(results[key])
        plot_data = plot_data[(plot_data.index % 10 == 0) | (plot_data.index == plot_data.index[-1])]

        plot_data.plot(ax=cur_ax, linewidth=0.4, logy=logy)
        cur_ax.set_title(f'{key}.', fontsize=15)

    # axes[0][0].set_ylim(-100, 20)
    # axes[0][1].set_ylim(0, 500)

    return fig, axes


def plot_mult_dyna_results(results, each=10, linewidth=0.4):
    # return results
    rows = int(np.ceil(len(list(results.items())[0][1][1]) / 2))
    fig, axes = plt.subplots(rows, 2)
    fig.set_size_inches(15, 5 * rows)

    for gamma in results:
        # print(gamma)
        for num, key in enumerate(results[gamma][1]):
            # print(key)
            logy = False if num == 0 else True
            cur_ax = axes[num // 2][num % 2]

            plot_data = pd.Series(results[gamma][1][key]).rolling(each, min_periods=1).mean()
            # plot_data = plot_data[(plot_data.index % each == 0)
            #                       | (plot_data.index == plot_data.index[-1])
            #                       | (plot_data.index < 10)]

            plot_data.plot(ax=cur_ax, linewidth=linewidth, logy=logy, label=gamma)
            cur_ax.set_title(f'{key}.', fontsize=15)
            cur_ax.legend()

    # axes[0][0].set_ylim(-100, 20)
    # axes[0][1].set_ylim(0, 500)

    return fig, axes


def viz_casino_policy(env, agent, second_best=None, Q=False):
    if not second_best:
        second_best = []
    results = {}
    starting_strategy = {}
    for cas_idx, cas in enumerate(env.casinos):
        s = (env.starting_money, cas_idx, 0, ())

        results[cas_idx] = {}
        if Q:
            if s in agent.Q:
                casino_value = max(agent.Q[s].values())
            else:
                casino_value = 0

            best_action = agent.P[s]
        else:
            casino_value = agent.V[s]
            best_action = agent.P[s]

        starting_strategy[cas_idx] = {'value': casino_value, 'best_first_action': best_action}

    starting_strategy = (
        pd.DataFrame(starting_strategy)
        .T
        .sort_values('value', ascending=False)
        .reset_index()
        .rename(columns={'index': 'cas_idx'})
    )

    for num, row in starting_strategy.iterrows():
        cas_idx = row.cas_idx
        if Q:
            second_best = starting_strategy.iloc[num+1:]['cas_idx'].tolist() if num < starting_strategy.shape[0] - 1 else []
        else:
            second_best = [starting_strategy.iloc[num + 1].cas_idx] if num < starting_strategy.shape[0] - 1 else []
        print(second_best)
    # for row, num in starting_strategy.iterrows():
    #     cas_idx = 1
        casino_policy = {}
        casino_value = {}
        for money in range(100, env.money_lim, 100):
            casino_policy[money] = {}
            casino_value[money] = {}
            for game_num in range(0, 20):

                s = (money, cas_idx, game_num, (*second_best,))
                if Q:
                    right_prefixes_states = list(filter(lambda x: x[:3] == (money, cas_idx, game_num), list(agent.C_out.keys())))

                    seconds_stats = {}
                    for right_state in right_prefixes_states:
                        seconds_stats[right_state] = 0
                        for a in agent.C_out[right_state]:
                            for outcome in agent.C_out[right_state][a]:
                                seconds_stats[right_state] += 1
                    if len(seconds_stats) == 0:
                        most_frequest_seconds = []
                    else:
                        most_frequest_seconds = max(seconds_stats.items(), key=lambda x: x[1])[0][-1]

                    # print(most_frequest_seconds)
                    # print(seconds_stats)
                    s = (money, cas_idx, game_num, (*most_frequest_seconds,))
                    # print(s)
                    if s in agent.Q:
                        casino_policy[money][game_num] = (
                            int(agent.P.get(s, 0)),
                            round(max(agent.Q[s].values())),
                            0 if s not in agent.C_out else sum(agent.C_out[s].get(agent.P[s], {0: 0}).values())
                        )
                    else:
                        casino_policy[money][game_num] = ('?', '?', '?')
                        # raise ValueError(s)
                else:
                    casino_policy[money][game_num] = (
                        int(agent.P.get(s, -10)),
                        round(agent.V.get(s, -10)),
                    )
                # casino_value[money][game_num] = round(agent.V.get((money, cas_idx, game_num, (*second_best,)), -10), 2)
        results[cas_idx]['P'] = pd.DataFrame(casino_policy).dropna()
        results[cas_idx]['V'] = pd.DataFrame(casino_value).dropna()
    return results, starting_strategy



import numpy as np
import functools
import itertools


def build_possible_left_casinos_set(current_casino, casinos_num):
    casinos_left = [x for x in range(casinos_num) if x != current_casino]
    possible_lens_of_rest = range(1, len(casinos_left) + 1)
    possible_rest_sets = [list(itertools.combinations(casinos_left, x)) for x in possible_lens_of_rest]
    return list(functools.reduce(lambda x, y: x + y, possible_rest_sets)) + [()]  # flatten


def correct_table(orig_table, decay, games_num):
    table = np.copy(orig_table).astype(float)
    if table.shape[0] == 1:  # Only losing
        return table
    else:
        win_row = table[0, :]
        lose_rows = table[1:, :]
        total_decay = decay * games_num
        if total_decay > lose_rows[:, 0].sum():
            raise ValueError('the games_num is too big')

        while total_decay > 0:
            lowest_prob = lose_rows[-1, 0]
            if lowest_prob <= total_decay:
                lose_rows[-1, 0] = 0
                total_decay -= lowest_prob
                win_row[0] += lowest_prob
                lose_rows = lose_rows[:-1, :]
            else:
                lose_rows[-1, 0] -= total_decay
                win_row[0] += total_decay
                total_decay = 0

        return np.vstack([win_row, lose_rows])


class Casinos:
    def __init__(self, casinos_num=5, money_lim=1000, starting_money=100, lose_bias=0.5, win_base_mult=1):
        self.win_dict = {
             1: [win_base_mult] + [2 * win_base_mult] + [3 * win_base_mult],
             2: [win_base_mult] + [2 * win_base_mult] * 2,
             3: [win_base_mult] + [2 * win_base_mult],
             4: [win_base_mult] * 3 + [2 * win_base_mult],
             5: [win_base_mult] * 4 + [2 * win_base_mult],
             6: [win_base_mult]
            }
        self.seed = 14
        self.money_lim = money_lim
        self.casinos_num = casinos_num
        self.lose_bias = int(lose_bias * 10)

        self.casinos = self.build_random_casinos(seed=self.seed)
        self.S = self.generate_states()
        self.starting_money = starting_money
        self.P = {}
        self.s = (0, -1, -1, (-1, ))

        self.build_mdp()

    def build_random_casinos(self, seed=14):
        np.random.seed(seed)
        casinos = []

        for casino in range(self.casinos_num):
            casino = np.array([[np.random.choice(
                range(self.lose_bias, self.lose_bias + 4), p=(0.4, 0.3, 0.2, 0.1)), 0]]
            )  # chance of losing
            while casino[:, 0].sum() < 10:
                current_prob_sum = (casino[:, 0].sum())
                win_chance = np.random.randint(1, 10 - current_prob_sum + 1)
                prize = np.random.choice(self.win_dict[win_chance])
                casino = np.vstack([casino, [[round(win_chance, 2), prize]]])
            casinos.append((casino, np.random.choice([0.2, 0.2, 0.3, 0.3, 0.4])))

        return casinos

    def generate_states(self):
        states = [(0, -1, -1, (-1, ))]  # final state
        for money in np.arange(100, self.money_lim, 100):
            for current_casino in range(len(self.casinos)):
                possible_decays = (10 - self.casinos[current_casino][0][0, 0]) / self.casinos[current_casino][1]
                for number_of_games in range(int(possible_decays) + 1):
                    for left_casinos in build_possible_left_casinos_set(current_casino, self.casinos_num):
                        states.append((money, current_casino, number_of_games, left_casinos))
        return states

    def build_mdp(self):
        print('Building MDP')
        self.P = {}
        length_S = len(self.S)

        for num, state in enumerate(self.S):
            print(f'\r{num}/{length_S}', end='')
            money, cur_casino_num, games_played, casinos_left = state

            if money == 0:
                self.P[tuple(state)] = {-100: [(1.0, (0, -1, -1, (-1,)), money, True), ]}
                continue

            table, decay = self.casinos[cur_casino_num]
            corrected_table = correct_table(table, decay, games_played)

            # Initialize actions
            possible_actions = {-100: [(1.0, (0, -1, -1, (-1,)), money-self.starting_money, True)]}  # leave

            # Change the casino
            for change_to_casino in casinos_left:
                # clear the new casino from list of possible new ones
                new_casinos_left = tuple([x for x in casinos_left if x != change_to_casino])
                # Generating new state
                next_state = (money, change_to_casino, 0, new_casinos_left)
                # Describe the reward and the probability
                possible_actions[-change_to_casino] = [(1.0, next_state, 0, False)]

            # Making bets
            for bet in range(100, money + 100, 100):
                possible_actions[bet] = []
                for prob, mult in corrected_table.tolist():
                    next_money = int(((money - bet) + bet * mult))

                    # If we reach the money limit
                    if next_money >= self.money_lim:
                        next_state = (0, -1, -1, (-1,))  # go to the final state
                        possible_actions[bet].append((  # get all the money we won
                            round(prob / 10, 2), next_state, next_money-self.starting_money, True
                        ))

                    # if we lose all our money
                    elif next_money == 0:
                        next_state = (0, -1, -1, (-1,))  # go to the final state
                        possible_actions[bet].append((
                            round(prob / 10, 2), next_state, next_money-self.starting_money, True
                        ))

                    else:
                        next_state = (next_money, cur_casino_num, games_played + 1, casinos_left)

                        # if the victory probability decayed to the zero we dont increment decay
                        if next_state not in self.S:
                            next_state = (next_money, cur_casino_num, games_played, casinos_left)

                        # Just in case we check if the state exists
                        assert next_state in self.S, f'{next_state}'

                        possible_actions[bet].append((
                            round(prob / 10, 2), next_state, 0, False
                        ))

                # Checking that each action has 1 probability distributed between possible outcomes
                assert sum(sum(map(lambda t: t[0], x)) for x in possible_actions.values()) == len(possible_actions),\
                    possible_actions

            self.P[tuple(state)] = possible_actions

    def reset(self):
        casinos_idx = range(0, len(self.casinos))
        start_casino = np.random.choice(casinos_idx)
        left_casinos = tuple(x for x in casinos_idx if x != start_casino)
        starting_state = (self.starting_money, start_casino, 0, left_casinos)
        assert starting_state in self.S, starting_state
        self.s = starting_state
        return self.s

    def step(self, a):
        np.random.seed(None)
        possible_outcomes = self.P[self.s][a]
        outcomes_probs = [x[0] for x in possible_outcomes]

        chosen_index = np.random.choice(len(possible_outcomes), p=outcomes_probs)
        prob, new_state, r, done = possible_outcomes[chosen_index]
        self.s = new_state

        return new_state, r, done, {}

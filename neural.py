import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from timeit import default_timer as timer
from datetime import timedelta

from torchinfo import summary

import matplotlib.pyplot as plt
import numpy as np

import random
import math
from collections import namedtuple, deque
from itertools import count

from take2 import GameState, Player, GameController, GameView, DummyActor, Actor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ResidualBlock(nn.Module):
    def __init__(self, classifications, internal_size):
        super(ResidualBlock, self).__init__()

        self.internal_size = internal_size
        self.classifications = classifications

        self.layer1 = nn.Linear(internal_size, internal_size)
        self.layer2 = nn.Linear(internal_size, internal_size)

        self.layer3 = nn.Linear(internal_size, (internal_size // 2) * 3)
        self.dropout1 = nn.Dropout(0.5)

        self.layer4 = nn.Linear((internal_size // 2) * 3, classifications)
        self.layer5 = nn.Linear(classifications, classifications)
        self.layer6 = nn.Linear(classifications, internal_size * 2)

        self.layer7 = nn.Linear(internal_size * 2, internal_size)

    def forward(self, x):
        x = F.relu(x + self.layer1(x))
        x = F.relu(x + self.layer2(x))
        x = F.relu(self.layer3(x))

        x = self.dropout1(x)
        x = F.sigmoid(self.layer4(x))
        x = F.softmax(self.layer5(x))

        x = F.relu(self.layer6(x))
        return F.relu(self.layer7(x))

class ConcatenationBlock(nn.Module):
    def __init__(self, internal_size):
        super(ConcatenationBlock, self).__init__()
        self.internal_size = internal_size

        self.layer = nn.Linear(self.internal_size * 2, self.internal_size)

    def forward(self, x):
        return F.relu(self.layer(x))

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, internal_size=256, classifications=16, feature_amt=4):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, internal_size // 2)
        self.layer2 = nn.Linear(internal_size // 2, internal_size)

        self.residual_amt = feature_amt
        self.classification_count = classifications
        self.res_block = []
        self.concat_block = []

        for i in range(self.residual_amt):
            self.res_block.append(ResidualBlock(classifications, internal_size).cuda())
            self.concat_block.append(ConcatenationBlock(internal_size).cuda())

        self.layer3 = nn.Linear(internal_size * self.residual_amt, internal_size)
        self.layer4 = nn.Linear(internal_size, internal_size // 2)
        self.layer5 = nn.Linear(internal_size // 2, n_actions)

    def forward(self, x):
        # Input
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        res = []
        for i in range(self.residual_amt):
            p = self.res_block[i].forward(x)

            q = torch.cat((x, p), 1)
            x = self.concat_block[i].forward(q)

            res.append(x.clone())

        r = torch.cat(res, 1)
        x = F.relu(self.layer3(r))
        x = F.relu(self.layer4(x))
        return self.layer5(x)


BATCH_SIZE = 512

# Larger gamma -> Future Reward is more important
GAMMA = 0.999

# What amount of sampled actions are random ... and at what rate does this decay?
EPS_START = 0.9
EPS_END = 0.0001
EPS_DECAY = 16384

# Target Net Update Rate
TAU = 0.01

# Base Learning Rate
LR = 0.01

device = 'cuda'

g_state = GameState()
g_state.players = [Player("Test")]
g_state.roll_dice_unsafe()

g_view = GameView(headless=True)

g_control = GameController(g_state, g_view, [DummyActor(0)])

n_actions = 64

state = g_state.to_vector(0)
n_observations = len(state)

internal_size = 256
classifications = 64
feature_amt = 6

policy_net = DQN(n_observations, n_actions, internal_size, classifications, feature_amt).to(device)
target_net = DQN(n_observations, n_actions, internal_size, classifications, feature_amt).to(device)
target_net.load_state_dict(policy_net.state_dict())

summary(ResidualBlock(classifications, internal_size), (1, internal_size))
summary(ConcatenationBlock(internal_size), (1, internal_size * 2))
summary(policy_net, (1, n_observations))

steps = 10
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(65536)

steps_done = 0


def get_max(opts, tensor):
    max = -1000000
    argmax = 0
    for opt in opts:
        prob = tensor[0][opt]
        if prob > max:
            max = prob
            argmax = opt

    return argmax

def select_action(s, c):
    global steps_done

    opts = c.get_valid_options()
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            t = get_max(opts, policy_net(s))
            return torch.tensor([[t]], device=device, dtype=torch.long)
    else:
        t = opts[random.randrange(0, len(opts))]
        return torch.tensor([[t]], device=device, dtype=torch.long)


episode_durations = []

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return False

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return True

coeff_punish = 0
coeff_chaos = 0
coeff_move = 0
coeff_hold = 1

def train(num_episodes):
    global BATCH_SIZE

    scores_sampler = []
    moves_sampler = []
    highest_sampler = []
    did_farkle_sampler = []
    delta_no_farkle_sampler = []
    did_hold_sampler = []

    start = timer()

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int(num_episodes/32), 1)
    for i_episode in range(num_episodes):

        g_control.get_current_state().accrue_total_score_and_reset_state()
        g_control.get_current_state().players[0].score = 0
        g_control.get_current_state().reset_move_num()

        dealt_score = g_control.get_current_state().get_hand_score()[0]
        finish_score = g_control.get_current_state().get_hand_score()[0]
        highest_score = g_control.get_current_state().get_hand_score()[0]

        total_reward = 0

        div_amt = 128

        moves = 0

        g_control.view.draw_turn(g_control.get_current_state())

        for t in count():
            finished = False
            state_old = g_control.get_current_state().make_copy()
            state_vec = torch.tensor(state_old.to_vector(0), dtype=torch.float32, device=device).unsqueeze(0)

            finish_score = state_old.get_round_score() + state_old.get_hand_score()[0]
            if finish_score > highest_score:
                highest_score = finish_score

            action = select_action(state_vec, g_control)

            g_control.do_action(action[0][0].item())

            observation = g_control.get_current_state().to_vector(0)
            state_new = g_control.get_current_state()

            reward = (state_new.get_round_score() + state_new.get_hand_score()[0] - finish_score) * coeff_move

            if action == 0:
                finish_score = state_old.get_round_score() + state_old.get_hand_score()[0]
                reward = finish_score/8000
                finished = True

            elif g_control.get_current_state().get_hand_score()[0] == 0:
                reward = -1
                finish_score = 0
                finished = True

            g_control.view.draw_move(action[0][0].item())
            g_control.view.draw_turn(g_control.get_current_state())

            moves += 1

            total_reward += reward
            reward = torch.tensor([reward], device=device)

            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state_vec, action, next_state, reward)

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)

            target_net.load_state_dict(target_net_state_dict)

            if finished:
                break

            scheduler.step()


        #string = tt.to_string(
        #    [[i_episode, round(i_episode/num_episodes * 100, 2), dealt_score, highest_score, finish_score if finish_score > 0 else "FARKLE!", moves, (finish_score - dealt_score) if (finish_score > dealt_score) else 0, round((1 - invalid_move_count / moves) * 100, 2), total_reward]],
        #    header=["Episode", "% Done", "Dealt Score", "Highest Score", "Finished Score", "Turn Count", "Delta Score", "% Valid Moves", "Total Reward"],
        #    style=tt.styles.ascii_thin_double,
        #)

        #print(string)

        scores_sampler.append(finish_score)
        moves_sampler.append(moves)

        highest_sampler.append(highest_score)

        if moves == 1:
            did_hold_sampler.append(1)
        else:
            did_hold_sampler.append(0)

        if finish_score == 0:
            did_farkle_sampler.append(1)
        else:
            delta_no_farkle_sampler.append(finish_score - dealt_score)
            did_farkle_sampler.append(0)

        if i_episode % 512 == 0:
            end = timer()
            t_delta = timedelta(seconds=end - start)
            print(round(i_episode/num_episodes * 100, 2), "%", "- [ t:", t_delta, "] -> lr:", scheduler.get_last_lr())

            if i_episode > 512:
                fig, axs = plt.subplots(3, 2, figsize=(16,12))
                axs[0, 0].plot(np.array(moving_average(scores_sampler, div_amt * 4)[::64]))
                axs[0, 0].set_title('Final Scores')

                axs[0, 1].plot(np.array(moving_average(highest_sampler, div_amt * 4)[::64]), 'tab:orange')
                axs[0, 1].set_title('Highest Scores')

                axs[1, 0].plot(np.array(moving_average(delta_no_farkle_sampler, div_amt * 4)[::64]), 'tab:green')
                axs[1, 0].set_title('Delta Scores [No Farkle]')

                axs[1, 1].plot(np.array(moving_average(moves_sampler, div_amt * 4)[::64]), 'tab:red')
                axs[1, 1].set_title('Move Count')

                axs[2, 0].plot(np.array(moving_average(did_farkle_sampler, div_amt * 4)[::64]), 'tab:green')
                axs[2, 0].set_title('Farkle Freq.')

                axs[2, 1].plot(np.array(moving_average(did_hold_sampler, div_amt * 4)[::64]), 'tab:red')
                axs[2, 1].set_title('Hold Freq.')

                div_amt += 128

                plt.savefig('plots/plot_' + str(t_delta).split(".")[0].replace(':', '-') + '.png')
                plt.close(fig)


class SimpleAIActor(Actor):
    def __init__(self, my_id):
        super().__init__(my_id)

    def get_action(self, state: GameState):
        cont = GameController(state, g_view, [DummyActor(0), DummyActor(1)])
        t = torch.tensor(state.to_vector(1), dtype=torch.float32, device=device).unsqueeze(0)
        return select_action(t, cont)[0][0].item()
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


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 512)

        # Residual Block 1
        self.layer4 = nn.Linear(512, 512)
        self.layer5 = nn.Linear(512, 512)
        self.layer6 = nn.Linear(512, 512)

        # Squeeze Layers
        self.layer7 = nn.Linear(512, 1024)

        self.dropout1 = nn.Dropout(0.2)

        self.layer8 = nn.Linear(1024, 128)
        self.layer9 = nn.Linear(128, 128)
        self.layer10 = nn.Linear(128, 1024)

        self.dropout2 = nn.Dropout(0.2)

        self.layer11 = nn.Linear(1024, 512)

        # Residual Block 2
        self.layer12 = nn.Linear(512, 512)
        self.layer13 = nn.Linear(512, 512)
        self.layer14 = nn.Linear(512, 512)

        self.layer15 = nn.Linear(512, 256)
        self.layer16 = nn.Linear(256, 128)
        self.layer17 = nn.Linear(128, n_actions)

    def forward(self, x):
        # Encode
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))

        # Residual Block 1
        x = F.relu(x + self.layer4(x))
        x = F.relu(x + self.layer5(x))
        x = F.relu(x + self.layer6(x))

        # Squeeze Architecture
        x = F.leaky_relu(self.layer7(x))
        x = self.dropout1(x)

        x = F.leaky_relu(self.layer8(x))
        x = F.softmax(self.layer9(x))
        x = F.leaky_relu(self.layer10(x))

        x = self.dropout2(x)
        x = F.leaky_relu(self.layer11(x))

        # Residual Block 2
        x = F.relu(x + self.layer12(x))
        x = F.relu(x + self.layer13(x))
        x = F.relu(x + self.layer14(x))

        # Decode
        x = F.relu(self.layer15(x))
        x = F.relu(self.layer16(x))
        return F.softmax(self.layer17(x))


BATCH_SIZE = 512

# Larger gamma -> Future Reward is more important
GAMMA = 0.4

# What amount of sampled actions are random ... and at what rate does this decay?
EPS_START = 0.9
EPS_END = 0.0001
EPS_DECAY = 12288

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

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

summary(policy_net, (64, 1, n_observations))

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

coeff_chaos = 1
coeff_move = 3
coeff_farkle = 1.5
coeff_hold = 2

def train(num_episodes):
    global BATCH_SIZE

    accumulated_score = 0

    scores_sampler = []
    moves_sampler = []
    delta_sampler = []
    highest_sampler = []

    start = timer()

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int(num_episodes/4), 2)
    for i_episode in range(num_episodes):

        g_control.get_current_state().accrue_total_score_and_reset_state()
        g_control.get_current_state().players[0].score = 0
        g_control.get_current_state().reset_move_num()

        dealt_score = g_control.get_current_state().get_hand_score()[0]
        finish_score = g_control.get_current_state().get_hand_score()[0]
        highest_score = g_control.get_current_state().get_hand_score()[0]

        hold_count = 0
        total_reward = 0

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
                reward = (dealt_score - finish_score) * coeff_hold
                reward = reward * moves**coeff_chaos - coeff_chaos * 100
                finished = True

            elif g_control.get_current_state().get_hand_score()[0] == 0:
                reward = -(finish_score * coeff_farkle) * (1 + coeff_chaos)**moves
                finish_score = 0
                finished = True

            g_control.view.draw_move(action[0][0].item())
            g_control.view.draw_turn(g_control.get_current_state())

            moves += 1

            reward = reward / 8000
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

        if i_episode % (num_episodes / 16) == 0:
            print(round(i_episode/num_episodes * 100, 2))
            print(scheduler.get_last_lr())
            end = timer()
            print("TIME: ", timedelta(seconds=end - start))

        accumulated_score += finish_score

        scores_sampler.append(finish_score)
        moves_sampler.append(moves)
        delta_sampler.append(finish_score - dealt_score)
        highest_sampler.append(highest_score)

    end = timer()
    print("TIME: ", timedelta(seconds=end - start))

    print("AVERAGE SCORE: ", accumulated_score / num_episodes)

    fig, axs = plt.subplots(2, 2)

    # axs[0, 0].plot(np.array(scores_sampler))
    # axs[0, 0].plot(np.array(moving_average(scores_sampler, 128)), 'tab:purple')
    axs[0, 0].plot(np.array(moving_average(scores_sampler, 1024)[::32]))
    axs[0, 0].set_title('Final Scores')

    # axs[0, 1].plot(np.array(highest_sampler), 'tab:orange')
    # axs[0, 1].plot(np.array(moving_average(highest_sampler, 128)), 'tab:purple')
    axs[0, 1].plot(np.array(moving_average(highest_sampler, 1024)[::32]), 'tab:orange')
    axs[0, 1].set_title('Highest Scores')

    # axs[1, 0].plot(np.array(delta_sampler), 'tab:green')
    # axs[1, 0].plot(np.array(moving_average(delta_sampler, 128)), 'tab:purple')
    axs[1, 0].plot(np.array(moving_average(delta_sampler, 1024)[::32]), 'tab:green')
    axs[1, 0].set_title('Delta Scores')

    # axs[1, 1].plot(np.array(moves_sampler), 'tab:red')
    # axs[1, 1].plot(np.array(moving_average(moves_sampler, 128)), 'tab:purple')
    axs[1, 1].plot(np.array(moving_average(moves_sampler, 1024)[::32]), 'tab:red')
    axs[1, 1].set_title('Move Count')

    plt.show()


class SimpleAIActor(Actor):
    def __init__(self, my_id):
        super().__init__(my_id)

    def get_action(self, state: GameState):
        cont = GameController(state, g_view, [DummyActor(0), DummyActor(1)])
        t = torch.tensor(state.to_vector(1), dtype=torch.float32, device=device).unsqueeze(0)
        return select_action(t, cont)[0][0].item()
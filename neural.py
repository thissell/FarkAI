import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


BATCH_SIZE = 1
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

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

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100)

steps_done = 0


def select_action(s):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(s).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randrange(0, 64)]], device=device, dtype=torch.long)


episode_durations = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
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


def train(num_episodes):
    for i_episode in range(num_episodes):
        state_old = g_control.get_current_state()
        state = torch.tensor(state_old.to_vector(0), dtype=torch.float32, device=device).unsqueeze(0)

        hold_count = 0
        turns = 0

        for t in count():
            action = select_action(state)

            if action == 0:
                hold_count += 1

            good = g_control.do_action(action[0][0].item())

            observation = g_control.get_current_state().to_vector(0)

            if good:
                reward = g_control.get_current_state().players[0].score + g_control.get_current_state().get_round_score() + g_control.get_current_state().get_hand_score()[0]
                reward = reward - (state_old.players[0].score + state_old.get_round_score() + state_old.get_hand_score()[0])
            else:
                reward = -1

            reward = torch.tensor([reward], device=device)
            done = g_state.get_hand_score()[0] == 0

            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)

            target_net.load_state_dict(target_net_state_dict)

            turns = t + 1
            if done:
                break

        print(i_episode, " : ", (g_control.get_current_state().players[0].score + g_control.get_current_state().get_round_score() + g_control.get_current_state().get_hand_score()[0]) / turns, ". % hold: ", hold_count / turns)
        g_control.get_current_state().players[0].score = 0


class SimpleAIActor(Actor):
    def __init__(self, my_id):
        super().__init__(my_id)

    def get_action(self, state: GameState):
        t = torch.tensor(state.to_vector(0), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return policy_net(t).max(1).indices.view(1, 1)[0][0].item()
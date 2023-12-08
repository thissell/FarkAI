import copy
import random
import collections

FULL_HAND = {0, 1, 2, 3, 4, 5}

UNDO_SIGNAL = -1
PRINT_SIGNAL = -2


class Player:
    def __init__(self, name):
        self.name = name
        self.score = 0


class GameState:
    def __init__(self):
        self.players = []
        self.dice = [0, 0, 0, 0, 0, 0]
        self.held = set()
        self.round_score = 0
        self.turn = 0
        self.move_num = 1

    def make_copy(self):
        copy = GameState()

        copy.players = [Player("this")]
        copy.dice = [i for i in self.dice]
        copy.held = self.held.copy()
        copy.round_score = self.round_score
        copy.turn = self.turn
        copy.move_num = self.move_num

        return copy

    def get_score(self, handset=None):
        if handset is None:
            handset = FULL_HAND
        arr = [self.dice[i] for i in handset]
        counts = collections.Counter(arr)
        k = [[], [], [], [], [], [], []]

        dice_set = set(arr)
        win_set = set()

        for die, count in counts.items():
            k[count].append(die)

        score = 0

        if len(k[6]) == 1:
            score = 3000
            win_set = win_set.union(set(k[6]))
        elif len(k[2]) == 3:
            score = 1500
            win_set = win_set.union(set(k[2]))
        elif len(k[1]) == 6:
            score = 1500
            win_set = win_set.union(set(k[1]))
        elif len(k[4]) == 1 and len(k[2]) == 1:
            score = 1500
            win_set = win_set.union(set(k[4]))
            win_set = win_set.union(set(k[2]))
        elif len(k[3]) == 2:
            score = 2500
            win_set = win_set.union(set(k[3]))
        else:
            if len(k[3]) == 1:
                if k[3][0] == 1:
                    score += 300
                else:
                    score += k[3][0] * 100

                win_set = win_set.union(set(k[3]))

            if len(k[4]) == 1:
                win_set = win_set.union(set(k[4]))
                score += 1000

            if len(k[5]) == 1:
                win_set = win_set.union(set(k[5]))
                score += 2000

            if 5 in k[1]:
                win_set.add(5)
                score += 50
        
            if 5 in k[2]:
                win_set.add(5)
                score += 100

            if 1 in k[1]:
                win_set.add(1)
                score += 100

            if 1 in k[2]:
                win_set.add(1)
                score += 200

        return score, win_set == dice_set

    # Returns to Score of the Hand
    def get_hand_score(self):
        return self.get_score(FULL_HAND.difference(self.held))
    
    # Returns the full Round Score
    def get_round_score(self):
        return self.round_score
    
    def get_current_player(self) -> Player:
        return self.players[self.turn]

    def inc_move_num(self):
        self.move_num += 1

    def reset_move_num(self):
        self.move_num = 1
    
    def to_vector(self, player_id):
        h_score = (self.get_hand_score()[0])/8000
        r_score = (self.get_round_score())/8000
        move_num = 1 / self.move_num

        d = ""
        for i in range(0, 5):
            d = d + str(self.dice[i] - 1)

        q = "{0:b}".format(int(d,6)).zfill(16)

        v = []
        for i in q:
            v.append(float(i))

        for i in range(0, 5):
            if i in self.held:
                v.append(1)
            else:
                v.append(0)

        v.append(h_score)
        v.append(r_score)
        v.append(move_num)

        return v

    # Goes to the Next Turn
    def next_turn(self):
        self.turn = (self.turn + 1) % len(self.players)

    # Puts the given dice into the held set
    def hold_dice(self, tomove: set[int]):
        hand = FULL_HAND.difference(self.held)

        if self.held.intersection(tomove) != set():
            return False
        
        move_score, all_score = self.get_score(tomove)
        if not all_score:
            return False

        self.round_score += move_score
        
        hand_left = hand.difference(tomove)
        if len(hand_left) == 0:
            self.held = set()
        else:
            self.held = self.held.union(tomove)

        return True

    def can_hold_dice(self, tomove: set[int]):
            if self.held.intersection(tomove) != set():
                return False

            move_score, all_score = self.get_score(tomove)
            if not all_score:
                return False

            return True

    # Rolls any Dice given
    def roll_dice_unsafe(self, roll_list: set[int] = None):
        if roll_list is None:
            roll_list = FULL_HAND

        for i in roll_list:
            self.dice[i] = random.randint(1, 6)
        return
    
    def accrue_total_score_and_reset_state(self):
        if self.get_hand_score()[0] != 0:
            self.players[self.turn].score += self.get_round_score() + self.get_hand_score()[0]
        
        self.round_score = 0
        self.held = set()
        self.roll_dice_unsafe()


class GameView:
    def __init__(self, headless):
        self.headless = headless

    def draw_line(self):
        if self.headless:
            return
        
        print("-------------------------------")

    def draw_move(self, move: int):
        if self.headless:
            return
        
        print("made move: ", end='')
        if move == 0:
            print("hold")
        else:
            q = "{0:b}".format(move - 1).zfill(6)
            for i,v in enumerate(q[::-1]):
                if v == '1':
                    print(i + 1, end='')

            print(" [" + str(move) + "]")
        return

    def draw_turn(self, state: GameState):
        if self.headless:
            return

        print(state.get_current_player().name + "'s turn:")
        print("held: ", end='')
        for i, v in enumerate(state.dice):
            if i in state.held:
                print(str(v) + " ", end='')
            else:
                print("  ", end='')

        print(" | held score: " + str(state.get_round_score()))

        print("hand: ", end='')
        for i, v in enumerate(state.dice):
            if i not in state.held:
                print(str(v) + " ", end='')
            else:
                print("  ", end='')

        print(" | hand score: " + str(state.get_hand_score()[0]))

        self.draw_line()

        return
    
    def draw_leaderboard(self, state: GameState):
        if self.headless:
            return
        
        print("Leaderboard:")
        for pl in state.players:
            print(pl.name + ": " + str(pl.score))

        self.draw_line()
        return
    
    def draw_farkle(self):
        if self.headless:
            return

        print("Farkle!")
        return


class Actor:
    def __init__(self, my_id):
        self.my_id = my_id

    def get_action(self, state: GameState):
        return 0


class PlayerActor(Actor):
    def __init__(self, my_id):
        super().__init__(my_id)

    # Returns an Action
    def get_action(self, state: GameState):
        g = input("action: ")

        if g == 'undo':
            return UNDO_SIGNAL
        
        if g == 'print':
            return PRINT_SIGNAL

        if g == 'hold':
            return 0
        else:
            g_safe = {'1', '2', '3', '4', '5', '6'}.intersection(set([i for i in g]))
            m_arr = set([int(i) - 1 for i in g_safe])

            out = 1
            for i in m_arr:
                out += 2 ** i

            return out


class SimpleHoldActor(Actor):
    def __init__(self, my_id):
        super().__init__(my_id)

    def get_action(self, state: GameState):
        return 0


class DummyActor(Actor):
    def __init__(self, my_id):
        super().__init__(my_id)

    def get_action(self, state: GameState):
        return 0


class GameController:
    def __init__(self, state: GameState, view: GameView, actors: list[Actor]):
        self.log = [state]
        self.view = view
        self.actors = actors

        if len(actors) != len(state.players):
            raise Exception("There must be a 1-1 relation between actors and players.")
    
    def get_current_state(self):
        return self.log[len(self.log) - 1]
    
    def clear_log(self):
        self.log = [self.log[-1]]
    
    def new_state(self):
        new_state = copy.deepcopy(self.get_current_state())
        self.log.append(new_state)
    
    def undo_state(self):
        self.log.pop()

    def begin_state(self):
        state = self.get_current_state()
        state.roll_dice_unsafe()

    def get_is_valid(self, act):
        state = self.get_current_state()

        if act == 1:
            return False

        if not act == 0:
            num_set = set()
            q = "{0:b}".format(act - 1).zfill(6)
            for i, v in enumerate(q):
                if v == '1':
                    num_set.add(5 - i)

            if not state.can_hold_dice(num_set):
                return False

        return True

    def get_valid_options(self):
        opts = []
        for i in range(64):
            if self.get_is_valid(i):
                opts.append(i)

        return opts

    def do_action(self, act):
        state = self.get_current_state()

        if act == UNDO_SIGNAL:
            self.log.pop()
            self.log.pop()

        if act == PRINT_SIGNAL:
            print([k.to_vector(0) for k in self.log])

        if act == 1:
            return False

        if act == 0:
            state.accrue_total_score_and_reset_state()
            state.next_turn()
            self.view.draw_line()
            self.view.draw_leaderboard(state)
        else:
            num_set = set()
            q = "{0:b}".format(act - 1).zfill(6)
            for i, v in enumerate(q):
                if v == '1':
                    num_set.add(5 - i)

            if state.hold_dice(num_set):
                if state.held == set():
                    state.roll_dice_unsafe()
                else:
                    state.roll_dice_unsafe(FULL_HAND.difference(state.held))
            else:
                return False

        self.get_current_state().inc_move_num()
        return True

    def loop_instance(self):
            self.new_state()
            state = self.get_current_state()
            self.view.draw_turn(state)

            before = state.players[state.turn].score + state.get_round_score() + (state.get_hand_score()[0])/2

            if state.get_hand_score()[0] == 0:
                self.view.draw_farkle()
                state.accrue_total_score_and_reset_state()
                state.next_turn()
                self.view.draw_line()
                self.view.draw_leaderboard(state)
                return

            act = self.actors[state.turn].get_action(state)
            self.view.draw_move(act)

            self.do_action(act)

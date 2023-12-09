from take2 import Player, GameState, GameView, GameController, PlayerActor
from neural import train, SimpleAIActor

train(8192)

actor1 = PlayerActor(0)
player1 = Player("Dumb")

actor2 = SimpleAIActor(1)
player2 = Player("Dumber")

state = GameState()
state.players = [player1, player2]

view = GameView(headless=False)
controller = GameController(state, view, [actor1, actor2])

controller.begin_state()

while True:
    controller.loop_instance()
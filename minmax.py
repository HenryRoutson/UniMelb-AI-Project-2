
MAX_DEPTH = 9
START_STATE = 0

DEBUG = True
C = 0.01 # from Upper Confidence Bound formula

# These two numbers should increase together
ITERATIONS = 100
EXPLORE_MIN = 13



from typing import Optional
import random

WinsAndGames = tuple[int, int] # TODO this should be something like average move or like top 3


# define simple game for testing
"""

Action is adding a number
State is a number
Player positive wins if the number if more positive
Negative if negative

"""




# ================================================================================
# config 


Player = str
PLAYER1 = "positive"
PLAYER2 = "negative"


State = int
Action = int

def heuristic(state : State, action : Action, player : Player) -> float :
  # used to pick which value to expand
  # this is much better than expanding randomly

  # TODO in actual game implimentation, make this the number of columns and rows that the color is in

  if player == PLAYER1 :
    return action #* random.random()
  else :
    return -action


def isStateWin(state : State) -> Optional[Player] :
  if state > 5 : return PLAYER1
  if state < -5 : return PLAYER2
  return None
  
def tieBreaker(state : State) -> Player :

  if state > 0 :
    return PLAYER1
  else :
    return PLAYER2


def getActionsFromState(state : State, player : Player) -> list[Action] :
  return [2, 1, -1, -2] 

def applyActionToState(state : State, action : Action, player : Player) -> State :
  return state + action

def rolloutStrategy(state : State, player: Player) :
  action = random.choice(getActionsFromState(state, player))
  return action









# ================================================================================
# impliment functions




class GameTree : # / node

  def __init__(self, children : list, winProp : WinsAndGames, action : Optional[Action]) -> None:
    self.children : list[GameTree] = children
    self.winProp : WinsAndGames = winProp # win proportion
    self.action = action 
    # state is derived, as it takes too much space


def printTree(tree : GameTree, state : Optional[State], toIndent = 100, indent = 0) :
  if indent > toIndent : return
  print("    "*indent + "action : " + str(tree.action) + ", winprop :" + str(tree.winProp) + "fract" + str(tree.winProp[0] / (tree.winProp[1] + 0.01))[1:4] + ", state : " + str(state)) # TODO add state
  for t in tree.children :
    tmpState = state
    if state != None and t.action :
      tmpState = applyActionToState(state, t.action)
    printTree(t, tmpState, indent=(indent + 1), toIndent=toIndent)

Children = list[GameTree]
Path = list[GameTree]

# functions =====


def reversePlayer(player : Player) -> Player :
  if player == PLAYER1 : return PLAYER2
  if player == PLAYER2 : return PLAYER1
  assert(False)

def whosMoveFromDepth(depth : int, playing : Player) -> Player :

  # tree root has no action and has moves for the player deciding where to move

  if (depth % 2 == 1) : 
    return playing
  else : 
    return reversePlayer(playing)




# ================================================================================
# impliment min max





# implimentation psuedo code
# https://www.youtube.com/watch?v=l-hh51ncgDI

def minMax(player : Player, fromState : State, isFirstMove : bool, toDepth : int = 3) -> Action :

  

  return bestAction













# ================================================================================
# call code

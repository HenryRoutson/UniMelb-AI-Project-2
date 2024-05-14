
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

def heuristic(state : State, player : Player) -> float :
  # used to pick which value to expand
  # this is much better than expanding randomly

  # TODO in actual game implimentation, make this the number of columns and rows that the color is in

  if player == PLAYER1 :
    return state #* random.random()
  else :
    return -state


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
  x = [2, 1, -1, -2]
  random.shuffle(x)
  return x


def applyActionToState(state : State, action : Optional[Action], whosMove : Player) -> State :

  if action == None : return state
  return state + action

def rolloutStrategy(state : State, player: Player) :
  action = random.choice(getActionsFromState(state, player))
  return action



# ================================================================================
# impliment functions


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



INF = float("inf")


def min_max(playing_player : Player, toDepth : int, state : State = START_STATE) :


  def min_max_helper(isMax : bool, toDepth : int, state : State, depth : int, whosMove : Player, alpha : float, beta : float)  -> tuple[Optional[Action], float] :

        (best_action, best_score) = (None, -INF if isMax else INF)
        min_or_max_f = max if isMax else min

        for action in getActionsFromState(state, whosMove) :
          result = min_max_sub(toDepth = toDepth, state = state, depth = depth + 1, action=action)
          (action, score) = result

          if (min_or_max_f(score, best_score) != best_score) :
            (best_action, best_score) = result

          # Alpha beta pruning
          if isMax :
            alpha = max(alpha, score)
          else :
            beta = min(beta, score)

          if (beta <= alpha) : break
          # 

        return (best_action, best_score)


  def min_max_sub(toDepth : int, state : State = START_STATE, action : Optional[Action] = None, depth : int = 1, alpha : float = -INF, beta : float = INF) -> tuple[Optional[Action], float] :
    print("depth :", depth)

    # calculate new state
    whosMove = whosMoveFromDepth(depth, playing_player)
    state = applyActionToState(state, action, whosMove)

    # base case
    if depth == toDepth : 
      return (action, heuristic(state, playing_player))

    # min or max case
    isMax = (whosMove == playing_player)
    result = min_max_helper(isMax=isMax, toDepth=toDepth, state=state, depth=depth, whosMove=whosMove, alpha= alpha, beta= beta)

    #
    return result


  return min_max_sub(toDepth=toDepth, state=state, action=None, depth=1, alpha=-INF, beta=INF)













# ================================================================================
# call code

print(min_max(toDepth=4, playing_player=PLAYER1))

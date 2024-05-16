
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

def heuristic(stateBefore : State, stateAfter : State, action : Action, player : Player) -> float :
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
  x = [1, -1] 
  #random.shuffle(x)
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

"""


def min_max(playing_player : Player, toDepth : int, state : State = START_STATE) :


  def min_max_helper(isMax : bool, toDepth : int, state : State, depth : int, whosMove : Player, alpha : float, beta : float)  -> tuple[Optional[Action], float] :

        (best_action, best_score) = (None, -INF if isMax else INF)
        min_or_max_f = max if isMax else min

        for action in getActionsFromState(state, whosMove) :
          result = min_max_sub(toDepth = toDepth, state = state, depth = depth + 1, action=action)
          (action, score) = result

          print(result, depth)

          if (min_or_max_f(score, best_score) != best_score) :
            (best_action, best_score) = result



          # Alpha beta pruning
          if isMax :
            alpha = max(alpha, score)
          else :
            beta = min(beta, score)

          if (beta <= alpha) : break
          # 


        print()
        return (best_action, best_score)


  def min_max_sub(toDepth : int, state : State, action : Optional[Action] = None, depth : int = 1, alpha : float = -INF, beta : float = INF) -> tuple[Optional[Action], float] :


    # calculate new state
    whosMove = whosMoveFromDepth(depth, playing_player)
    stateAfter = applyActionToState(state, action, whosMove)

    # base case
    if depth == toDepth : 
      assert(action) # depth shouldn't equal 1
      return (action, heuristic(stateAfter=stateAfter, stateBefore=state, action=action, player=playing_player))

    # min or max case
    isMax = (whosMove == playing_player)
    result = min_max_helper(isMax=isMax, toDepth=toDepth, state=stateAfter, depth=depth, whosMove=whosMove, alpha= alpha, beta= beta)

    if whosMove == PLAYER1 : assert(result[0] == 2)
    if whosMove == PLAYER2 : 
      if not (result[0] == -2) :

        print()
        print( result)
        
        print("ERR")
        exit()

        # TODO fix this bug

    #
    return result


  return min_max_sub(toDepth=toDepth, state=state, action=None, depth=1, alpha=-INF, beta=INF)




"""

##############


def whosMoveFromDepth(depth : int, playing : Player) -> Player :

  # tree root has no action and has moves for the player deciding where to move

  if not (depth % 2 == 1) : 
    return playing
  else : 
    return reversePlayer(playing)






def heuristic(state : State, player : Player) -> float :
  # used to pick which value to expand
  # this is much better than expanding randomly

  # TODO in actual game implimentation, make this the number of columns and rows that the color is in

  if player == PLAYER1 :
    return state #* random.random()
  else :
    return -state
  


def min_max(playing_player : Player, toDepth : int, state : State, depth : int = 0, alpha : float = -INF, beta : float = INF) -> tuple[list[Action], float] :
  # TODO clould indent other min for perf
  
  # alpha represents best move for max player, so will only update on their move
  # beta represents best move for min player, so will only update on their move

  # the best next move for alpha is the worst move for beta, hence 




  if depth == toDepth : 
    return ([], heuristic(state = state, player= playing_player))
  
  # TODO
  """
  if isStateWin(state) != None :
    return ([], INF if playing_player == isStateWin(state) else -INF)
  """

  whosMove = whosMoveFromDepth(depth, playing_player)

  best_action : Optional[Action] = None

  if playing_player == whosMove :

    best_value : float = -INF
    for action in getActionsFromState(state, playing_player) :
        new_state = applyActionToState(state=state, action=action, whosMove=whosMove)
        nextActions, cur_value = min_max(playing_player=playing_player, depth=depth + 1, toDepth=toDepth, state=new_state, alpha=alpha, beta=beta)

        if cur_value == max([cur_value, best_value]) :
          best_value = cur_value
          best_action = action

        alpha = max(alpha, cur_value)
        if beta <= alpha : break
          
  else :
        
    best_value : float = INF
    for action in getActionsFromState(state, playing_player) :
        new_state = applyActionToState(state=state, action=action, whosMove=whosMove)
        nextActions, cur_value = min_max(playing_player=playing_player, depth=depth + 1, toDepth=toDepth, state=new_state, alpha=alpha, beta=beta)

        if cur_value == min([cur_value, best_value]) :
          best_value = cur_value
          best_action = action

        beta = min(beta, cur_value)
        if beta <= alpha : break

  # 

  assert(best_action != None) 

  lst = [best_action]
  lst.extend(nextActions)
  return (lst, best_value)




# RETRY











# ================================================================================
# call code

print(min_max(toDepth=2, playing_player=PLAYER1, state=START_STATE))

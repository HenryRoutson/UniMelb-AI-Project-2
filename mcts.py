
MAX_DEPTH = 9
START_STATE = 0
EXPLORE_MIN = 13
DEBUG = False
C = 0.1 # from Upper Confidence Bound formula
ITERATIONS = 1000

from typing import Optional
import random

WinsAndGames = tuple[int, int]


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


def getActionsFromState(state : State) -> list[Action] :
  return [2, 1, -1, -2] # TODO test this and check win rate

def applyActionToState(state : State, action : Action) -> State :
  return state + action

def rolloutStrategy(state : State, player: Player) :
  action = random.choice(getActionsFromState(state))
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

def scoreFromwinProp(winProp : WinsAndGames) -> float : # TODO need to seperate uncertainty from win rate
  # TODO use actual formula
  # can make more complicated with uncertainty from lower number of games
  return winProp[0] / winProp[1]

def updateWinsAndGames(winProp : WinsAndGames, didWin : bool) -> WinsAndGames :
  return (winProp[0] + didWin, winProp[1] + 1)

def getMinOrMaxFromChildren(parent : GameTree, isMax) -> int :
  children = parent.children

  assert(children != [])

  # uncertainty
  e = list(enumerate(children))
  random.shuffle(e)
  for i, c in e :
    if c.winProp[1] < EXPLORE_MIN :  # explore unexplored
      return i


  # win probability
  scores = list(map(lambda child : UCB(Parent_n=parent, n=child), children))
  #scores = list(map(lambda child : scoreFromwinProp(child.winProp), children))

  if isMax : getValue = max(scores)
  else : getValue = min(scores)

  max_index = scores.index(getValue)
  return max_index

def getMinMaxPath(tree : GameTree, isMaxFirst : bool, state : State) -> tuple[list[GameTree], State] :
  

  path : list[GameTree] = [tree] # path indexes 
  while tree.children != [] :

    if (tree.action) :
      state = applyActionToState(state, tree.action)

    next_i = getMinOrMaxFromChildren(tree, isMaxFirst)
    next = tree.children[next_i]
    path.append(next)
    tree = next
    isMaxFirst = not isMaxFirst 

  if DEBUG :
    print("Path")
    for c in path :
      print(c.action)
    print("Path end")
    
  return path, state


def scoreFromTree(x : GameTree) :
  return scoreFromwinProp(x.winProp)

def rolloutSim(state : State, whosMove : Player, depth : int) -> Player :

  while depth != MAX_DEPTH :

    action = rolloutStrategy(state, whosMove)
    state = applyActionToState(state, action)
    maybeSomeoneWon = isStateWin(state)

    if maybeSomeoneWon != None : 
      someoneWon = maybeSomeoneWon
      return someoneWon

    whosMove = reversePlayer(whosMove)
    depth += 1

  return tieBreaker(state)


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

def makeMoveWith(initState : State, tree : GameTree, player: Player) -> GameTree :
  isMaxFirst = True

  # 1 Selection (min max)
  path, leafState = getMinMaxPath(tree, isMaxFirst, initState)
  depth = len(path)
  whosMove = whosMoveFromDepth(depth=depth, playing=player)

  # 2 Expansion (add a single node)
  leafNode = path[-1]
  leafActions = getActionsFromState(leafState)
  for action in leafActions : 
    leafNode.children.append(GameTree([], (0, 0), action))
  
  def heuristicFromChild(child : GameTree) :
    assert(child.action)
    return heuristic(state=leafState, action=child.action, player=whosMove)

  leafNode.children.sort(key=heuristicFromChild, reverse=True)

  if DEBUG :
    print("Children actions ranked")
    for t in leafNode.children: print(t.action)
    print()

  path.append(leafNode.children[0])

  # 3 Simulation (rollout)


  # 3.1 derive state
  action = path[-1].action
  assert(action)
  state = applyActionToState(leafState, action) 

  # 3.2 simluate rollout
  whoWon = rolloutSim(state, whosMove, depth=depth)
  didWin = (whoWon == player)

  # 4 Back-propagation (update win and games values)
  for curNode in path :
    curNode.winProp = updateWinsAndGames(curNode.winProp, didWin)

  return tree

import math

def U(t : GameTree) -> float : return float(t.winProp[0])
def N(t : GameTree) -> float : return float(t.winProp[1])

def UCB(Parent_n : GameTree, n : GameTree) :
  assert(n in Parent_n.children)
  return (U(n) / N(n)) + C * math.sqrt(math.log(N(Parent_n), 2) / N(n))

def mcts(player : Player, fromState : State, iterations = 5000) -> Action :

  gameTree = GameTree([], (0,0), None) # starting node
  for _ in range(iterations) :
    gameTree = makeMoveWith(fromState, gameTree, player)
    if DEBUG :
      print("Tree")
      printTree(gameTree, fromState, toIndent=3)

  nodes, endState = getMinMaxPath(gameTree, True, fromState)
  bestAction = nodes[1].action # 1 to ignore start node
  assert(bestAction)
  return bestAction



# ================================================================================
# call code

print(mcts(fromState=START_STATE, iterations=ITERATIONS, player=PLAYER1))




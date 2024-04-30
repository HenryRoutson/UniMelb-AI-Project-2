
# https://pythontutor.com/visualize.html


PLAYER1 = "positive"
PLAYER2 = "negative"



START_STATE = 0

from typing import Optional
import random

WinsAndGames = tuple[int, int]
Player = str

# define simple game for testing
State = int
Action = int
class GameTree : # / node

  def __init__(self, children : list, winProp : WinsAndGames, action : Optional[Action]) -> None:
    self.children : list[GameTree] = children
    self.winProp : WinsAndGames = winProp # win proportion
    self.action = action 
    # state is derived, as it takes too much space


def printTree(tree : GameTree, state : Optional[State], toIndent = 100, indent = 0) :
  if indent > toIndent : return
  print("    "*indent + "action  :" + str(tree.action) + " winprop :" + str(tree.winProp) + "state :" + str(state)) # TODO add state
  for t in tree.children :
    tmpState = None
    if state and t.action :
      tmpState = applyActionToState(state, t.action)
    printTree(t, tmpState, indent=(indent + 1), toIndent=toIndent)



Children = list[GameTree]
Path = list[GameTree]

# functions =====

def scoreFromwinProp(winProp : WinsAndGames) -> float :
  # can make more complicated with uncertainty from lower number of games
  if winProp[0] == 0 : return 2.0 # exlore unexplored
  return winProp[1] / winProp[0]

def updateWinsAndGames(winProp : WinsAndGames, didWin : bool) -> WinsAndGames :
  return (winProp[0] + didWin, winProp[1] + 1)

def getMinOrMaxFromChildren(children : Children, isMax) -> int :

  assert(children != [])
  scores = list(map(lambda child : scoreFromwinProp(child.winProp), children))

  if isMax : getValue = max(scores)
  else : getValue = min(scores)

  max_index = scores.index(getValue)
  return max_index

def getMinMaxPath(tree : GameTree, isMaxFirst : bool, state : State) -> tuple[list[GameTree], State] :

  path : list[GameTree] = [tree] # path indexes 
  while tree.children != [] :

    if (tree.action) :
      state = applyActionToState(state, tree.action)

    isMaxFirst = not isMaxFirst # may have mixed up negative
    next_i = getMinOrMaxFromChildren(tree.children, isMaxFirst)
    next = tree.children[next_i]
    path.append(next)
    tree = next
    

  return path, state

def getActionsFromState(state : State) -> list[Action] :
  return [1, -1, -2] # TODO test this and check win rate

def applyActionToState(state : State, action : Action) -> State :
  return state + action

def rolloutStrategy(state : State, player: Player) :
  # TODO test doing this random and improving this

  action = random.choice(getActionsFromState(state))
  #action = int(isMaxFirst)
  
  return action

MAX_DEPTH = 9

def rolloutSim(state : State, isMaxFirst: bool) -> Player :

  # TODO while testing
  """
  depth = 0
  while depth != MAX_DEPTH :

    action = rolloutStrategy(state, isMaxFirst)
    state = applyActionToState(state, action)

    optionalWin = isStateWin(state, isMaxFirst)
    if optionalWin != None : return optionalWin
    
    isMaxFirst = not isMaxFirst
  
    depth += 1
  """

  return tieBreaker(state)

def makeMoveWith(initState : State, tree : GameTree, player: Player) -> GameTree :
  isMaxFirst = True

  # 1 Selection (min max)
  path, leafState = getMinMaxPath(tree, isMaxFirst, initState)
  depth = len(path)

  # 2 Expansion (add a single node)
  leafNode = path[-1]
  leafActions = getActionsFromState(leafState)
  for action in leafActions :
    leafNode.children.append(GameTree([], (0, 0), action))
  
  path.append(random.choice(leafNode.children))

  # 3 Simulation (rollout)
 
  # 3.1 derive state
  action = path[-1].action
  assert(action)
  state = applyActionToState(leafState, action) # TODO where does this action come from

  # 3.2 simluate rollout
  whoWon = rolloutSim(state, isMaxFirst)
  didWin = (whoWon == player)

  # 4 Back-propagation (update win and games values)
  for curNode in path :
    curNode.winProp = updateWinsAndGames(curNode.winProp, didWin)

  return tree

def isStateWin(state : State, isMax : bool) -> Optional[Player] :
  if isMax :
    if state > 5 : return PLAYER1
  else :
    if state < -5 : return PLAYER2

  return None
  
def tieBreaker(state : State) -> Player :

  if state > 0 :
    return PLAYER1
  else :
    return PLAYER2


# call code =====



gameTree = GameTree([], (0,0), None)

for _ in range(20000) :
  gameTree = makeMoveWith(START_STATE, gameTree, PLAYER1)
  print("Tree")
  printTree(gameTree, START_STATE, toIndent=3)


# TODO now you can see the best move to make


"""
gameTree = GameTree([ GameTree([GameTree([], (1,0), -1)], (1,0), -1),  GameTree([], (0,1), 1)], (0,0), None)
printTree(gameTree)
"""




# TODO need to test min max
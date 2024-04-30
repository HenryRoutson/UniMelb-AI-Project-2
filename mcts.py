
# https://pythontutor.com/visualize.html

PLAYER = False

from typing import Optional
import random

WinsAndGames = tuple[int, int]

# define simple game for testing
State = int
Action = int
class GameTree : # / node

  def __init__(self, children : list, winProp : WinsAndGames, action : Optional[Action]) -> None:
    self.children : list[GameTree] = children
    self.winProp : WinsAndGames = winProp # win proportion
    self.action = action 
    # state is derived, as it takes too much space


def printTree(tree : GameTree, indent = 0) :
  print("    "*indent + "action  :" + str(tree.action) + " winprop :" + str(tree.winProp))
  for t in tree.children :
    printTree(t, indent + 1)



Children = list[GameTree]
Path = list[GameTree]

# functions =====

def scoreFromwinProp(winProp : WinsAndGames) -> float :
  # can make more complicated with uncertainty from lower number of games
  if winProp[0] == 0 : return 1.0
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

def getMinMaxPath(tree : GameTree, isMaxFirst : bool) -> list[int] :

  path_i : list[int] = [] # path indexes
  while tree.children != [] :

    next_i = getMinOrMaxFromChildren(tree.children, isMaxFirst)
    next = tree.children[next_i]
    path_i.append(next_i)
    tree = next
    isMaxFirst = not isMaxFirst

  return path_i

def getActionsFromState(state : State) -> list[Action] :
  return [1, -1]

def applyActionToState(state : State, action : Action) -> State :
  return state + action

def rolloutStrategy(state : State, isMaxFirst: bool) :
  # TODO test doing this random and improving this

  action = random.choice(getActionsFromState(state))
  #action = int(isMaxFirst)
  
  return action

MAX_DEPTH = 9

def rolloutSim(state : State, isMaxFirst: bool) -> bool :

  depth = 0
  while depth != MAX_DEPTH :

    action = rolloutStrategy(state, isMaxFirst)
    state = applyActionToState(state, action)

    optionalWin = isStateWin(state, isMaxFirst)
    if optionalWin != None : return optionalWin
    
    isMaxFirst = not isMaxFirst
  
    depth += 1

  return tieBreaker(state)

def makeMoveWith(initState : State, tree : GameTree, isMaxFirst: bool) -> GameTree :

  # 1 Selection (min max)
  path_i = getMinMaxPath(tree, isMaxFirst)
  depth = len(path_i)

  # 2 Expansion (add a single node)

  leafState = initState
  leafNode = tree

  for index in path_i :

    if (leafNode.action) :
      leafState = applyActionToState(leafState, leafNode.action)
    leafNode = leafNode.children[index]

  action = rolloutStrategy(leafState, isMaxFirst)
  leafNode.children.append(
    GameTree([], (0, 0), action)
  )

  # 3 Simulation (rollout)
 
  # 3.1 derive state
  state = applyActionToState(leafState, action)

  # 3.2 simluate rollout
  whoWon = rolloutSim(state, isMaxFirst)
  didWin = (whoWon == PLAYER)

  # 4 Back-propagation (update win and games values)
  curNode = tree
  for i in range(len(path_i)) :
    curNode.winProp = updateWinsAndGames(curNode.winProp, didWin)
    curNode = curNode.children[path_i[i]]

  return tree

def isStateWin(state : State, isMax : bool) -> Optional[bool] :
  if isMax :
    if state > 5 : return True
  else :
    if state < -5 : return False

  return None
  
def tieBreaker(state : State) -> bool :
  return state > 0 

# call code =====

gameTree = GameTree([], (0,0), None)

for _ in range(10) :
  gameTree = makeMoveWith(0, gameTree, True)
printTree(gameTree)


"""
gameTree = GameTree([ GameTree([GameTree([], (1,0), -1)], (1,0), -1),  GameTree([], (0,1), 1)], (0,0), None)
printTree(gameTree)
"""

# TODO
# fix backprop with simple line tree
# increasing branching factor
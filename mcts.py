
# https://pythontutor.com/visualize.html




from typing import Optional
import random

WinsAndGames = tuple[int, int]

# define simple game for testing
State = int
Action = int


class GameTree : # / node

  def __init__(self, children : list, winProp : WinsAndGames, action : Action) -> None:
    self.children : list[GameTree] = children
    self.winProp : WinsAndGames = winProp # win proportion
    self.action = action 
    # state is derived, as it takes too much space

Children = list[GameTree]
Path = list[GameTree]


def scoreFromwinProp(winProp : WinsAndGames) -> float :
  # can make more complicated with uncertainty from lower number of games
  return winProp[1] / winProp[0]


def updateWinsAndGames(winProp : WinsAndGames, didWin : bool) -> WinsAndGames :
  return (winProp[1] + didWin, winProp[1] + 1)

def updatePathWinsAndGames(path : Path, didWin : bool) -> Path : 
  for i in range(len(path)) :
    path[i].winProp = updateWinsAndGames(path[i].winProp, didWin)
  return path


def getMinOrMaxFromChildren(children : Children, isMax) -> int :

  assert(children != [])

  scores = list(map(lambda child : scoreFromwinProp(child.winProp), children))

  if isMax :
    getValue = max(scores)
  else :
    getValue = min(scores)

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


def playout(tree : GameTree) -> bool :
  didWin = random.choice([True, False]) # TODO
  return didWin



# TODO simulate

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

  while depth != MAX_DEPTH :

    action = rolloutStrategy(state, isMaxFirst)
    state = applyActionToState(state, action)

    optionalWin = isStateWin(state, isMaxFirst)
    if optionalWin != None : 
      return optionalWin
    
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

    leafState = applyActionToState(leafState, leafNode.action)
    leafNode = leafNode.children[index]
  

  # TODO make more intelligent moves
  action = rolloutStrategy(leafState, isMaxFirst)
  leafNode.children.append(
    GameTree([], (0, 0), action)
  )

  # TODO apply action to state

  # 3 Simulation (rollout)
 
  # 3.1 derive state
  state = applyActionToState(leafState, action)

  # 3.2 simluate rollout
  result = rolloutSim(state, isMaxFirst)

  # 4 Back-propagation (update win and games values)
  path_i.append(0)
  for index in path_i :
    # update values
    pass 
    # TODO
  
  return tree

  
def isStateWin(state : State, isMax : bool) -> Optional[bool] :
  if isMax :
    if state > 5 : return True
  else :
    if state < -5 : return False

  return None
  
def tieBreaker(state : State) -> bool :
  return state > 0 
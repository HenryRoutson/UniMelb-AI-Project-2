
from typing import Optional
import random

WinsAndGames = tuple[int, int]

# define simple game for testing
State = int
Action = int


class GameTree : # / node

  def __init__(self, children : list, winProp : WinsAndGames, state : State) -> None:
    self.children : list[GameTree] = children
    self.winProp : WinsAndGames = winProp # win proportion
    self.state = state

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


def getMinOrMaxFromChildren(children : Children, isMax) -> GameTree :

  assert(children != [])

  scores = list(map(lambda child : scoreFromwinProp(child.winProp), children))

  if isMax :
    getValue = max(scores)
  else :
    getValue = min(scores)

  max_index = scores.index(getValue)
  
  return children[max_index]
  

def getMinMaxPath(tree : GameTree, isMaxFirst : bool) -> Path :

  path : Path = []
  while tree.children != [] :

    next = getMinOrMaxFromChildren(tree.children, isMaxFirst)
    path.append(next)
    tree = next
    isMaxFirst = not isMaxFirst

  return path


def playout(tree : GameTree) -> bool :
  didWin = random.choice([True, False]) # TODO
  return didWin



# TODO simulate








def getActionsFromState(state : State) -> list[Action] :
  return [1, -1]

  
def applyActionToState(state : State, action : Action) -> State :
  return state + action

def makeMove(state : State) -> Action :

  actions = getActionsFromState(state)
  for action in actions :
    newState = applyActionToState(state, action)

  # TODO


  """

  1 Selection 
  2 Expansion 
  3 Simulation
  4 Back-propagation
  """


  return 1

  
def isStateWin(state : State, isPlayer1 : bool) -> bool :
  if isPlayer1 :
    return state > 5
  else :
    return state < -5
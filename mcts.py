
from typing import Optional
import random

WinsAndGames = tuple[int, int]
class GameTree : # / node

  def __init__(self, children : list, winProp : WinsAndGames) -> None:
    self.children : list[GameTree] = children
    self.winProp : WinsAndGames = winProp # win proportion

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



  



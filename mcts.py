
from typing import Optional


WinsAndGames = tuple[int, int]
class GameTree :

  def __init__(self, children : list, winProp : WinsAndGames) -> None:
    self.children : list[GameTree] = children
    self.winProp : WinsAndGames = winProp # win proportion

Children = list[GameTree]


def scoreFromwinProp(winProp : WinsAndGames) -> float :
  # can make more complicated with uncertainty from lower number of games
  return winProp[1] / winProp[0]


def updateWinsAndGames(winProp : WinsAndGames, didWin : bool) -> WinsAndGames :
  return (winProp[1] + didWin, winProp[1] + 1)

def updatePathWinsAndGames(path : list[GameTree], didWin : bool) -> list[GameTree] : 
  for i in range(len(path)) :
    path[i].winProp = updateWinsAndGames(path[i].winProp, didWin)
  return path




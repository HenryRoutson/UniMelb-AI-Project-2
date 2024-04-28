
from typing import Optional


WinsAndGames = tuple[int, int]
class GameTree :

  def __init__(self, children : list, data : WinsAndGames) -> None:
    self.children : list[GameTree] = children
    self.data : WinsAndGames = data


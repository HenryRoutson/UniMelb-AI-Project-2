
from typing import Optional


WinsAndGames = tuple[int, int]
class GameTree :

  def __init__(self, children : Optional[list], data : WinsAndGames) -> None:
    self.children : Optional[list[GameTree]] = children
    self.data : WinsAndGames = data


GameTree(None, (0, 0))
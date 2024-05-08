# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Action, PlaceAction, Coord
from a1 import Board
from collections import Counter

State = Board


MAX_DEPTH = 9
START_STATE = {} # empty board

DEBUG = False
C = 0.01 # from Upper Confidence Bound formula

# These two numbers should increase together
ITERATIONS = 75
EXPLORE_MIN = 13



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


Player = PlayerColor




def heuristic(state : State, action : Action, player : Player) -> float :
  # used to pick which value to expand
  # this is much better than expanding randomly

  # TODO in actual game implimentation, make this the number of columns and rows that the color is in

  return 0


def isStateWin(state : State) -> Optional[Player] :
  if not state.values().__contains__(PlayerColor.RED) : return PlayerColor.BLUE
  if not state.values().__contains__(PlayerColor.BLUE) : return PlayerColor.RED
  return None
  
def tieBreaker(state : State) -> Optional[Player] :

  counts = Counter(state.values())

  if counts[PlayerColor.BLUE.value] > counts[PlayerColor.RED.value] : return PlayerColor.BLUE
  if counts[PlayerColor.RED.value] > counts[PlayerColor.BLUE.value] : return PlayerColor.RED
  return None

def getActionsFromState(state : State) -> list[Action] :
  return [] # TODO

def applyActionToState(state : State, action : Action) -> State :
  return state # TODO

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
  if player == PlayerColor.RED : return PlayerColor.BLUE
  if player == PlayerColor.BLUE : return PlayerColor.RED
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

print()
print("best move for player : ")
print(mcts(fromState=START_STATE, iterations=ITERATIONS, player=PlayerColor.BLUE))








class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Tetress game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """

        # Below we have hardcoded two actions to be played depending on whether
        # the agent is playing as BLUE or RED. Obviously this won't work beyond
        # the initial moves of the game, so you should use some game playing
        # technique(s) to determine the best action to take.
        match self._color:
            case PlayerColor.RED:
                print("Testing: RED is playing a PLACE action")
                return PlaceAction(
                    Coord(3, 3), 
                    Coord(3, 4), 
                    Coord(4, 3), 
                    Coord(4, 4)
                )
            case PlayerColor.BLUE:
                print("Testing: BLUE is playing a PLACE action")
                return PlaceAction(
                    Coord(2, 3), 
                    Coord(2, 4), 
                    Coord(2, 5), 
                    Coord(2, 6)
                )

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after an agent has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        # There is only one action type, PlaceAction
        place_action: PlaceAction = action
        c1, c2, c3, c4 = place_action.coords

        # Here we are just printing out the PlaceAction coordinates for
        # demonstration purposes. You should replace this with your own logic
        # to update your agent's internal game state representation.
        print(f"Testing: {color} played PLACE action: {c1}, {c2}, {c3}, {c4}")




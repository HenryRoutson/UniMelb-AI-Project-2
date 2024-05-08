# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Action, PlaceAction, Coord

from typing import Optional
import random
import math

WinsAndGames = tuple[int, int]

State = dict[Coord, PlayerColor]
BOARD_SIZE = 11
BOARD_ITER = range(BOARD_SIZE)


MAX_DEPTH = 9
START_STATE = 0

DEBUG = False
C = 0.01

# These two numbers should increase together
ITERATIONS = 75
EXPLORE_MIN = 13

henryAction = int


class GameTree : # / node

  def __init__(self, children : list, winProp : WinsAndGames, action : Optional[henryAction]) -> None:
    self.children : list[GameTree] = children
    self.winProp : WinsAndGames = winProp # win proportion
    self.action = action
    # state is derived, as it takes too much space

Children = list[GameTree]
Path = list[GameTree]



def printTree(tree : GameTree, state : Optional[State], toIndent = 100, indent = 0) :
  if indent > toIndent : return
  print("    "*indent + "action : " + str(tree.action) + ", winprop :" + str(tree.winProp) + "fract" + str(tree.winProp[0] / (tree.winProp[1] + 0.01))[1:4] + ", state : " + str(state)) # TODO add state
  for t in tree.children :
    tmpState = state
    if state != None and t.action :
      tmpState = applyActionToState(state, t.action)
    printTree(t, tmpState, indent=(indent + 1), toIndent=toIndent)



def getActionsFromState(state : State) -> list[henryAction] :
    return [2, 1, -1, -2] 

def applyActionToState(state : State, action : henryAction) -> State :
    return state + action

def rolloutStrategy(state : State, color: PlayerColor) :
    action = random.choice(getActionsFromState(state))
    return action


def initial_game_state() -> State:
    """
    Returns the initial game state, which is an empty board.
    """
    initial_state = {}
    for x in BOARD_ITER:
        for y in BOARD_ITER:
            initial_state[Coord(x, y)] = None
    return initial_state



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
        self.game_state = initial_game_state()
        match color:
            case PlayerColor.RED:

                print("Testing: I am playing as RED")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")
        

        self.game_tree = GameTree([], (0, 0), None)  # Initialize an empty game tree


    def heuristic(self, state : State, action : henryAction, color : PlayerColor) -> float :
    # used to pick which value to expand
    # this is much better than expanding randomly

        if color == self._color :
            return action #* random.random()
        else :
            return -action



    def isStateWin(self, state : State) -> Optional[PlayerColor] :
        if state > 5 : return self._color
        if state < -5 :
            match self._color:
                case PlayerColor.RED:
                    return PlayerColor.BLUE
                case PlayerColor.BLUE:
                    return PlayerColor.RED

        return None
  
    def tieBreaker(self, state : State) -> PlayerColor :

        if state > 0 :
            return self._color
        else :
            match self._color:
                case PlayerColor.RED:
                    return PlayerColor.BLUE
                case PlayerColor.BLUE:
                    return PlayerColor.RED



    def scoreFromwinProp(self, winProp : WinsAndGames) -> float : # TODO need to seperate uncertainty from win rate
        # TODO use actual formula
        # can make more complicated with uncertainty from lower number of games
        return winProp[0] / winProp[1]
    


    def updateWinsAndGames(self, winProp : WinsAndGames, didWin : bool) -> WinsAndGames :
        return (winProp[0] + didWin, winProp[1] + 1)
    


    def getMinOrMaxFromChildren(self, parent : GameTree, isMax) -> int :
        children = parent.children

        assert(children != [])

        # uncertainty
        e = list(enumerate(children))
        for i, c in e :
            if c.winProp[1] < EXPLORE_MIN :  # explore unexplored
                return i


        # win probability
        scores = list(map(lambda child : self.UCB(Parent_n=parent, n=child), children))
        #scores = list(map(lambda child : scoreFromwinProp(child.winProp), children))

        if isMax : getValue = max(scores)
        else : getValue = min(scores)

        max_index = scores.index(getValue)
        return max_index
    


    def getMinMaxPath(self, tree : GameTree, isMaxFirst : bool, state : State) -> tuple[list[GameTree], State] :
  

        path : list[GameTree] = [tree] # path indexes 
        while tree.children != [] :

            if (tree.action) :
                state = applyActionToState(state, tree.action)

            next_i = self.getMinOrMaxFromChildren(tree, isMaxFirst)
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
    

    def scoreFromTree(self, x : GameTree) :
        return self.scoreFromwinProp(x.winProp)

    def rolloutSim(self, state : State, whosMove : PlayerColor, depth : int) -> PlayerColor :

        while depth != MAX_DEPTH :

            action = rolloutStrategy(state, whosMove)
            state = applyActionToState(state, action)
            maybeSomeoneWon = self.isStateWin(state)

            if maybeSomeoneWon != None : 
                someoneWon = maybeSomeoneWon
                return someoneWon

            whosMove = self.reversePlayer(whosMove)
            depth += 1

        return self.tieBreaker(state)
    
    def reversePlayer(self, color : PlayerColor) -> PlayerColor :
        if color == PlayerColor.RED : return PlayerColor.BLUE
        if color == PlayerColor.BLUE : return PlayerColor.RED
        assert(False)

    def whosMoveFromDepth(self, depth : int, playing : PlayerColor) -> PlayerColor :

        # tree root has no action and has moves for the player deciding where to move

        if (depth % 2 == 1) : 
            return playing
        else : 
            return self.reversePlayer(playing)



    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """

        # Use MCTS to select the best action
        best_action = self.mcts(self._color, fromState=START_STATE, iterations=ITERATIONS)
        return best_action

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after an agent has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        # There is only one action type, PlaceAction
        if isinstance(action, PlaceAction):
            for coord in action.coords:
                self.game_state[coord] = color
                print(f"Testing: {color} played PLACE action: {self.game_state[coord]}")

        # Here we are just printing out the PlaceAction coordinates for
        # demonstration purposes. You should replace this with your own logic
        # to update your agent's internal game state representation.


    def makeMoveWith(self, initState : State, tree : GameTree, player: PlayerColor) -> GameTree :
        isMaxFirst = True

        # 1 Selection (min max)
        path, leafState = self.getMinMaxPath(tree, isMaxFirst, initState)
        depth = len(path)
        whosMove = self.whosMoveFromDepth(depth=depth, playing=player)

        # 2 Expansion (add a single node)
        leafNode = path[-1]
        leafActions = getActionsFromState(leafState)
        for action in leafActions : 
            leafNode.children.append(GameTree([], (0, 0), action))
        
        def heuristicFromChild(child : GameTree) :
            assert(child.action)
            return self.heuristic(state=leafState, action=child.action, color=whosMove)

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
        whoWon = self.rolloutSim(state, whosMove, depth=depth)
        didWin = (whoWon == player)

        # 4 Back-propagation (update win and games values)
        for curNode in path :
            curNode.winProp = self.updateWinsAndGames(curNode.winProp, didWin)

        return tree
    

    def U(self, t : GameTree) -> float : return float(t.winProp[0])
    def N(self, t : GameTree) -> float : return float(t.winProp[1])

    def UCB(self, Parent_n : GameTree, n : GameTree) :
        assert(n in Parent_n.children)
        return (self.U(n) / self.N(n)) + C * math.sqrt(math.log(self.N(Parent_n), 2) / self.N(n))


    def mcts(self, player : PlayerColor, fromState : State, iterations = 5000) -> henryAction :

        gameTree = GameTree([], (0,0), None) # starting node
        for _ in range(iterations) :
            gameTree = self.makeMoveWith(fromState, gameTree, player)
            if DEBUG :
                print("Tree")
                printTree(gameTree, fromState, toIndent=3)

        nodes, endState = self.getMinMaxPath(gameTree, True, fromState)
        bestAction = nodes[1].action # 1 to ignore start node
        assert(bestAction)
        return bestAction


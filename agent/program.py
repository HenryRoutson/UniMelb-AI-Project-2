







########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

# PART A

























# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part A: Single Player Tetress




from inspect import isclass
from itertools import groupby
from re import L

from referee.game import PlayerColor, Action, PlaceAction, Coord
from collections import Counter

from typing import Callable, Optional
from typing import NewType
from queue import PriorityQueue
from collections import defaultdict
import math
import copy
import time
import gc
from collections.abc import Iterable




from enum import Enum
from dataclasses import dataclass
from typing import Generator

# WARNING: Please *do not* modify any of the code in this file, as this could
#          break things in the submission environment. Failed test cases due to
#          modification of this file will not receive any marks. 
#
#          To implement your solution you should modify the `search` function
#          in `program.py` instead, as discussed in the specification.

BOARD_N = 11



@dataclass(frozen=True, slots=True)
class Vector2:
    """
    A simple 2D vector "helper" class with basic arithmetic operations
    overloaded for convenience.
    """
    r: int
    c: int

    def __lt__(self, other: 'Vector2') -> bool:
        return (self.r, self.c) < (other.r, other.c)
    
    def __hash__(self) -> int:
        return hash((self.r, self.c))
    
    def __str__(self) -> str:
        return f"Vector2({self.r}, {self.c})"

    def __add__(self, other: 'Vector2|Direction') -> 'Vector2':
        return self.__class__(self.r + other.r, self.c + other.c)

    def __sub__(self, other: 'Vector2|Direction') -> 'Vector2':
        return self.__class__(self.r - other.r, self.c - other.c)

    def __neg__(self) -> 'Vector2':
        return self.__class__(self.r * -1, self.c * -1)

    def __mul__(self, n: int) -> 'Vector2':
        return self.__class__(self.r * n, self.c * n)

    def __iter__(self) -> Generator[int, None, None]:
        yield self.r
        yield self.c

    def down(self, n: int = 1) -> 'Vector2':
        return self + Direction.Down * n
    
    def up(self, n: int = 1) -> 'Vector2':
        return self + Direction.Up * n
    
    def left(self, n: int = 1) -> 'Vector2':
        return self + Direction.Left * n
    
    def right(self, n: int = 1) -> 'Vector2':
        return self + Direction.Right * n


class Direction(Enum):
    """
    An `enum` capturing the four cardinal directions on the game board.
    """
    Down  = Vector2(1, 0)
    Up    = Vector2(-1, 0)
    Left  = Vector2(0, -1)
    Right = Vector2(0, 1)

    @classmethod
    def _missing_(cls, value: tuple[int, int]):
        for item in cls:
            if item.value == Vector2(*value):
                return item
        raise ValueError(f"Invalid direction: {value}")

    def __neg__(self) -> 'Direction':
        return Direction(-self.value)

    def __mul__(self, n: int) -> 'Vector2':
        return self.value * n

    def __str__(self) -> str:
        return {
            Direction.Down:  "[↓]",
            Direction.Up:    "[↑]",
            Direction.Left:  "[←]",
            Direction.Right: "[→]",
        }[self]

    def __getattribute__(self, __name: str) -> int:
        match __name:
            case "r":
                return self.value.r
            case "c":
                return self.value.c
            case _:
                return super().__getattribute__(__name)



def apply_ansi(
    text: str, 
    bold: bool = True, 
    color: str | None = None
):
    """
    Wraps some text with ANSI control codes to apply terminal-based formatting.
    Note: Not all terminals will be compatible!
    """
    bold_code = "\033[1m" if bold else ""
    color_code = ""
    if color == "r":
        color_code = "\033[31m"
    if color == "b":
        color_code = "\033[34m"
    return f"{bold_code}{color_code}{text}\033[0m"

def render_board(
    board: dict[Coord, PlayerColor], 
    ansi: bool = True
) -> str:
    """
    Visualise the Tetress board via a multiline ASCII string, including
    optional ANSI styling for terminals that support this.

    If a target coordinate is provided, the token at that location will be
    capitalised/highlighted.
    """
    output = ""
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            if board.get(Coord(r, c), None):
                color = board[Coord(r, c)]
                color = "r" if color == PlayerColor.RED else "b"
                text = f"{color}" 
                if ansi:
                    output += apply_ansi(text, color=color)
                else:
                    output += text
            else:
                output += "."
            output += " "
        output += "\n"
    return output



INF = float('inf')
Board = dict[Coord, PlayerColor]
Target = NewType('Target', Coord)
PlaceActionLst = list[PlaceAction]
Heuristic = Callable[[Board, PlaceActionLst, Target], int]
BOARD_SIZE = 11
BOARD_ITER = range(BOARD_SIZE)
MAX_PIECE_SIZE = 4
KEY_INDEX = 0   # coord
VALUE_INDEX = 1 # colour

Heuristic_value = tuple[int, int]



def squaresToPieces(numSquares : int) -> int :
  # return min number of pieces to create number of squares

  return math.ceil(numSquares / MAX_PIECE_SIZE) # if remainder, you still need another piece









def emptyPlacesCount(board : Board, index : int, isColumn : bool) -> int :
  # return the number of empty places in a row of column at an index
   
  row_func = lambda col : not Coord(r=index, c=col) in board.keys()
  col_func = lambda row : not Coord(r=row, c=index) in board.keys()

  f = None
  if isColumn :
     f = col_func
  else :
     f = row_func

  return sum(map(f, BOARD_ITER))



########################################################################################################################
########################################################################################################################
########################################################################################################################

# Code for removing filled lines








def fillColumnOrRowCompileTime(index : int, isColumn : bool) -> set[Coord] :
     
  if isColumn :

    def iterToColumnCoord(i : int) :
      return Coord(r=i, c=index)

    l = map(iterToColumnCoord, BOARD_ITER)

  else :

    def iterToRowCoord(i : int) :
      return Coord(r=index, c=i)

    l = map(iterToRowCoord, BOARD_ITER)

  return set(l)


FILLED_COLUMNS_AND_ROW_SETS = [
   [fillColumnOrRowCompileTime(i, False) for i in BOARD_ITER],
   [fillColumnOrRowCompileTime(i, True) for i in BOARD_ITER],
]



def fillColumnOrRow(index : int, isColumn : bool) -> set[Coord] :

  result = FILLED_COLUMNS_AND_ROW_SETS[isColumn][index]
  return result


def removeCoords(board : Board, coords : set[Coord]) -> Board :
    for key in coords :
      board.pop(key, None)
  
    return board


def removeRowOrColumnFromBoard(board : Board, index : int, isColumn : bool) -> Board:
  
  line : set[Coord] = fillColumnOrRow(index, isColumn)
  board = removeCoords(board, line) # sometimes not all coords will be present if row and column are eliminated at the same time
  return board

def checkAndRemoveColumnOrRowFilled(board : Board, index : int, isColumn : bool) -> tuple[Board, bool] :

  line : set[Coord] = fillColumnOrRow(index, isColumn)

  lineRemainder : set[Coord] = line - board.keys()
  isLineFilled : bool = len(lineRemainder) == 0

  if (isLineFilled) :
     board = removeCoords(board, line)

  return (board, isLineFilled) # board, didElim





def boardEliminateFilledRowsOrColumnsWrapper(board : Board) -> tuple[Board, bool] :
  
  fast = boardEliminateFilledRowsOrColumns(board)
  return fast

allCoordsSet = set()
for r in BOARD_ITER :
   for c in BOARD_ITER :
      allCoordsSet.add(Coord(r=r, c=c))


boardIterSet = set(BOARD_ITER)


def missingIndexes(s : set) -> set : 
  return boardIterSet - s

def missingCoords(coords : Iterable[Coord]) -> set[Coord] :
  return allCoordsSet - set(coords)


def columnsAndRowsFullyOccupied(board : Board) :

  # columns and rows which are occupied 
  # are not occupied by empty squares

  # find rows and columns occupied by one empty piece
  # and then take the inverse to find rows and columns not occupied by an empty piece

  unoccupiedCoords = missingCoords(board.keys())
  columns_with_emtpy, rows_with_empty = columnsAndRowsOccupied(unoccupiedCoords)

  columns_without_empty = missingIndexes(columns_with_emtpy)
  rows_without_empty = missingIndexes(rows_with_empty)

  return (columns_without_empty, rows_without_empty)



def boardEliminateFilledRowsOrColumns(board : Board) -> tuple[Board, bool] :

  columns, rows = columnsAndRowsFullyOccupied(board)
  didElim = (len(columns) != 0) or (len(rows) != 0)

  # remove from board
  for c in columns :
     board = removeRowOrColumnFromBoard(board, c, True)
  for r in rows : 
     board = removeRowOrColumnFromBoard(board, r, False)

  return board, didElim






########################################################################################################################
########################################################################################################################
########################################################################################################################


def wrappingIndexDistance(index : int, target : int) -> int:
  # distance between two values on wrapping number line of BOARD_SIZE
  
  abs_dif = abs(index - target)
  return min(abs_dif, BOARD_SIZE - abs_dif)






def getCoordsOfColour(board : Board, colour : PlayerColor) -> list[Coord] :


  items_of_place_colour = filter(lambda item : (item[VALUE_INDEX] == colour) ,list(board.items()))
  squares_of_place_colour : list[Coord] = list(map(lambda item : item[KEY_INDEX], items_of_place_colour))

  return squares_of_place_colour
  
  

def a_star_len(BOARD : Board, coords_from : list[Coord], index : int, isColumn : bool, PlayerColor : PlayerColor) -> int :
  tmp = a_star(BOARD, coords_from, index, isColumn, PlayerColor)
  if tmp == None : return -1
  else : return len(tmp)


def minSquaresOfColorToIndexRowOrColumn(board : Board, target_index : int, isColumn, PlaceColour : PlayerColor) -> int :
  # manhattan distance

  squares_of_place_colour : list[Coord] = getCoordsOfColour(board, PlaceColour)

  # TODO config here to use a* or simple manhattan
  #return a_star_len(board, squares_of_place_colour, target_index, isColumn)
  return a_star_heuristic(squares_of_place_colour, target_index, isColumn)


def minNumSquaresToDeleteTarget(board : Board, target : Target) :
   return min( \
    emptyPlacesCount(board, target.r, isColumn=False), \
    emptyPlacesCount(board, target.c, isColumn=True ), \
  )
   


def getLineIndexes(board : Board, index : int, isColumn : bool) -> list[int] :

  if isColumn :
    tmp = [i for i in BOARD_ITER if (coord := Coord(r=i, c=index)) if (coord in board.keys()) ] 
  else :
    tmp = [i for i in BOARD_ITER if (coord := Coord(r=index, c=i)) if (coord in board.keys()) ] 

  print(index, isColumn, tmp)
  return tmp

"""

# leave

def combine_sequential(lst):
  pass

  



assert(combine_sequential([1,2,5,6]) == [(1,2), (5,6)])
assert(combine_sequential([1,2,3,4,5,6]) == [(1,2,3,4,5,6)])
assert(combine_sequential([1,3,5]) == [(1), (3), (5)])

def gapSizeListFromLine(line : list[int]) :

  empty = set(BOARD_ITER) - set(line)
  # if 10 and zero then combine
  return list(map(len, combine_sequential(empty)))
  

assert(gapSizeListFromLine([0, 1, 2, 5, 7, 8 , 9, 10]) == [2, 2])   # 1 2 _ _ 5 _ _ 7 8 9 10 11

print("HERE")
print(gapSizeListFromLine([5, 7, 8 , 9, 10]) )
assert(gapSizeListFromLine([5, 7, 8 , 9, 10]) == [4, 1])   # _ _ _ _ _ 5 _ _ 7 8 9 10
assert(gapSizeListFromLine([1, 2, 3, 4, 5, 6, 7, 8, 9]) == [2])




def squaresToCompleteLine(board : Board, index : int, isColumn : bool) -> int :

  # TODO each gap requires at least 


  fillInCost = squaresToCompleteLine(board, index, isColumn)
  goAroundCost = 0

  # TODO multiply by 4

  
  print(render_board(board, None, ansi=True)) # remove for submission
  print(goAroundCost, index, isColumn)
  return fillInCost + goAroundCost


"""


def minSquareCostForRowOrColumn(board : Board, target : Coord, isColumn : bool, PlaceColour: PlayerColor) -> int :
  if isColumn : index = target.c
  else        : index = target.r

  empty_places_on_line = emptyPlacesCount(board, index, isColumn)
  
  tmp = minSquaresOfColorToIndexRowOrColumn(board, index, isColumn, PlaceColour)
  if (tmp == -1) : return -1
  squares_to_line = max(tmp - 1, 0) # -1 becuase if you have a square on the line that doesn't matter, it only needs to be next to the line
  min_num_squares = squares_to_line + empty_places_on_line

  return min_num_squares


def numSquaresToPieces(i : int) :
  return  math.ceil(i / 4) # 


def heuristic1(derived_board : Board, placeActionLst : PlaceActionLst, target : Target, PlaceColour: PlayerColor) -> Heuristic_value | None :

  ##################################################################################

  current_cost = placeActionLst.__len__()
  
  ##################################################################################

  # number of squares

  remaining_squares = min(
    minSquareCostForRowOrColumn(derived_board, target, False, PlaceColour),
    minSquareCostForRowOrColumn(derived_board, target, True, PlaceColour),
  )

  if (remaining_squares == -1) : return None

  underestimate_remaining_pieces = numSquaresToPieces(remaining_squares)

  ##################################################################################

  admissble = current_cost + underestimate_remaining_pieces
  not_admissble = current_cost + remaining_squares

  assert(admissble <= not_admissble)

  ##################################################################################

  return (admissble, not_admissble)


def isPieceDeleted(board : Board, target : Target) -> bool : 
  return not target in board
   # return minNumSquaresToDeleteTarget(board, target) == 0

def isSquareEmpty(coord : Coord, board : Board) -> bool :
  return not coord in board.keys()

def isPiecePlaceSquaresEmpty(place : PlaceAction, board : Board) -> bool :

  for c in list(place.coords) : 
    squareCoord : Coord = c
    if not isSquareEmpty(squareCoord, board) :
       return False
    
  return True





# HERE TODO
def coordSquareNeighbors(coord : Coord) -> list[Coord] :
  return [
    coord.__add__(Direction.Up),
    coord.__add__(Direction.Down),
    coord.__add__(Direction.Left),
    coord.__add__(Direction.Right)
  ]

def coordEmptySquareNeighbors(board : Board, coord : Coord) -> list[Coord] :
  return list(filter(lambda coord : coord not in board.keys() ,coordSquareNeighbors(coord)))


def overLap(place : PlaceAction ,coord : Coord) : 
  return coord in place.coords
  


def printBoardPlaceAction(placeActions : list[PlaceAction], PlaceColour : PlayerColor) :
  print(render_board(deriveBoardBruteForce({}, placeActions, PlaceColour)[0]))

def coordPlaceOptions(board : Board, around : Coord) -> Iterable[PlaceAction]:
  # all place actions around the around coord

  """
  if (not around in board) or board[around] != PLACE_COLOUR : 
    assert(False)
    return [] # placement may have been eliminated
  """

  # assert(board[around] == PLACE_COLOUR) # to connect other pieces, this needs to be the place colour

  """
  options = []

  for placement_adj_to_center in GENERATED_PIECE_PLACEMENTS :
      
      #assert(not overLap(placement_adj_to_center, CENTER))
      #assert(adjacentTo(placement_adj_to_center, CENTER))

      # this is used to move placeAction from being adjacent to center, to being adjacent to the empty square
      placement_adj_to_around = offsetPlaceAction(placement_adj_to_center, around, CENTER) 
      #assert(not overLap(placement_adj_to_around, around))
      #assert(adjacentTo(placement_adj_to_around, around))

      if isPiecePlaceSquaresEmpty(placement_adj_to_around, board):
        options.append(placement_adj_to_around)

  #assert(len(set(options)) == len(options))

  """


  def filterByBoardSquaresBeingEmpty(placemnt : PlaceAction) :
     return isPiecePlaceSquaresEmpty(placemnt, board)

  def makeGeneratedAdjacentToAround(placement : PlaceAction) :
     return offsetPlaceAction(placement, around, CENTER) 

  placements_adj_to_around = map(makeGeneratedAdjacentToAround, GENERATED_PIECE_PLACEMENTS)
  options = filter(filterByBoardSquaresBeingEmpty, placements_adj_to_around)

  return options
  
  

def placeActionsFromBoard(board : Board, PlaceColour : PlayerColor) -> Iterable[PlaceAction]:
  # return a list of all pieces you can place connecting to a placeAction

  neighbors = set()

  for coord in getCoordsOfColour(board, PlaceColour) :
    neighbors.update(coordPlaceOptions(board, coord))

  return neighbors



def qcopy(board : Board) -> Board :
  newBoard : Board = dict()
  
  for (key, value) in board.items() :
    newBoard[key] = value

  return newBoard




def removeColumnsAndRowsOnCoord(coord : Coord , board : Board) -> tuple[Board, bool]  :
   
    didElim = False

    board, didElimThisTime = checkAndRemoveColumnOrRowFilled(board, coord.c, True)
    didElim = max(didElim, didElimThisTime)

    board, didElimThisTime = checkAndRemoveColumnOrRowFilled(board, coord.r, False)
    didElim = max(didElim, didElimThisTime)

    return (board, didElim)


def deriveBoard(original_board : Board, placeActionLst : PlaceActionLst, PlaceColour : PlayerColor) -> tuple[Board, bool] :
  """
  This function is hyperoptimised,
  so don't worry if it looks needlessly complicated
  
  """

  board = qcopy(original_board)

  didElim = False
  for place in placeActionLst :
     for coord in place.coords :
        
        if coord in board.keys() :
           board, didElimThisTime = removeColumnsAndRowsOnCoord(coord, board)
           if didElimThisTime : didElim = True
           
        board[coord] = PlaceColour

  (board, didElimThisTime) = boardEliminateFilledRowsOrColumnsWrapper(board)
  if didElimThisTime : didElim = True

  return (board, didElim)



def deriveBoardBruteForce(original_board : Board, PlaceActionLst : PlaceActionLst, PlaceColour : PlayerColor) -> tuple[Board, bool] :

  board = qcopy(original_board)

  didElim = False
  
  for place in PlaceActionLst :
     for coord in place.coords :
                
        board[coord] = PlaceColour
        (board, didElimThisTime) = boardEliminateFilledRowsOrColumns(board)
        if didElimThisTime : didElim = True
        
  return (board, didElim)






# bruh is there a way i don't have to do this, i hate you can't just add methods in a seperate file

class PlaceActionListWrapper():
  
    placeLst : PlaceActionLst

    def __init__(self, placeLst):
      self.placeLst = placeLst
    
    def __eq__(self, other):
        return self

    def __ne__(self, other):
        return self

    def __lt__(self, other):
        return self

    def __le__(self, other):
        return self

    def __gt__(self, other):
        return self

    def __ge__(self, other):
        return self



# int first to sort priority
pq_Type = tuple[Heuristic_value, PlaceActionListWrapper]


PQ_TYPE_HEURISTIC_VALUE_INDEX = 0
PQ_TYPE_PLACE_ACTION_WRAPPER_INDEX = 1


def pq_Type_get_PlaceActionLst(qp_type :pq_Type) -> list[PlaceAction] :
    
  wrapper : PlaceActionListWrapper = qp_type[PQ_TYPE_PLACE_ACTION_WRAPPER_INDEX]
  ret : PlaceActionLst = wrapper.placeLst
  return ret

def heuristic_search(BOARD : Board, TARGET : Target, PlaceColour : PlayerColor) ->  PlaceActionLst | None :
  """
  Adapted from https:#en.wikipedia.org/wiki/A*_search_algorithm
  BOARD and TARGET are contant
  """

  pq : PriorityQueue[pq_Type] = PriorityQueue()

  def addToPQ_returnTrueIfSolution(placeActions : PlaceActionLst, initial_derived_board : Board) -> bool :
     # this derived board includes an extra piece

    neighbor = placeActions[-1]
    [derived_board_plus, didElimLine] =  deriveBoard(initial_derived_board, [neighbor], PlaceColour) # include neighbor
    

    """
    # more performant to leave out
    froze = frozenset(derived_board_plus.items())
    if (froze in reached) : return False # already reached
    reached.add(froze)
    """

    

    # shouldn't already be in pq
    if not placeActions in (map(lambda x : x[1].placeLst,pq.queue)):

      cost : Heuristic_value | None = heuristic1(derived_board_plus, placeActions, TARGET, PlaceColour)
      if (cost == None) : return False # can't reach

      to_put = (cost, PlaceActionListWrapper(placeActions))
      pq.put(to_put)

    if didElimLine and isPieceDeleted(derived_board_plus, TARGET) :

      print("SOLUTION")
      print(render_board(derived_board_plus, TARGET, ansi=True))
      
      return True
    
    return False



    

  # Add all neighbors of existing squares 

  for coord in BOARD.keys() :
    if BOARD[coord] == PlaceColour :
      for n in coordPlaceOptions(BOARD, coord) :
        neighbor : PlaceAction = n
        #assert(adjacentTo(neighbor, coord))

        placeAction = [neighbor]
        if addToPQ_returnTrueIfSolution(placeAction, BOARD) :
          return placeAction
        




  while not pq.empty() :

    current_with_heuristic : pq_Type = pq.get(block=False)
    current : PlaceActionLst = pq_Type_get_PlaceActionLst(current_with_heuristic)

    #print("Place search")
    #print("\ncurrent cost   : " + str(len(current)))
    #print("heuristic cost : " + str(current_with_heuristic[PQ_TYPE_HEURISTIC_VALUE_INDEX])) 

    [derived_board, didElimLine] = deriveBoard(BOARD, current, PlaceColour)

    #print(render_board(derived_board, TARGET, ansi=True)) # remove for submission


    for neighbor in placeActionsFromBoard(derived_board, PlaceColour) :

        next : PlaceActionLst = current + [neighbor]
        if addToPQ_returnTrueIfSolution(next, derived_board) :
          return next


  return None



# TODO write render board function


a_star_pq_Type = tuple[int, list[Coord]]

# use derive board to take into account elimination of rows and columns




def coordToPlaceAction(coord : Coord ) -> PlaceAction :
  return PlaceAction(coord, coord, coord, coord)

def a_star_heuristic(coord_list : list[Coord], index : int, isColumn : bool) :
  return wrappingIndexDistance(getIndex(coord_list[-1], isColumn), index)

def deriveBoardForAStar(board : Board, current : list[Coord], PlaceColour : PlayerColor) :
  return deriveBoard(board, [coordToPlaceAction(coord) for coord in current], PlaceColour)


def getIndex(coord : Coord, isColumn : bool) :
  if isColumn : return coord.c
  return coord.r

def coordListToHashable(list : list[Coord]) :
  return ' '.join([(str(x.r) + str(x.c)) for x in list])

def a_star(BOARD : Board, coords_from : list[Coord], index : int, isColumn : bool, PlaceColour : PlayerColor) ->  list[Coord] | None :
  """
  Adapted from https:#en.wikipedia.org/wiki/A*_search_algorithm
  """

  #print("A* A* A* A* A*")


  pq : PriorityQueue[a_star_pq_Type] = PriorityQueue()
  reached = set()


  def addToPQ_returnTrueIfSolution(coord_list : list[Coord], initial_derived_board : Board) -> bool :
     # this derived board includes an extra piece

    last = coord_list[-1]
    if (last in reached) : return False

    reached.add(last)
    
    if getIndex(last, isColumn) == index : # adjacent to coord_to

      #print("SOLUTION")
      # TODO render board
      
      return True

    if (not coord_list in (map(lambda x : x[1], pq.queue))) :
      cost : int = a_star_heuristic(coord_list, index, isColumn)
      to_put = (cost, coord_list)
      pq.put(to_put)

    return False

  # Add all neighbors of existing squares 

  for c in coords_from:
    for start_move in coordEmptySquareNeighbors(BOARD, c) :
     if addToPQ_returnTrueIfSolution([start_move], BOARD) :
        return [start_move]
        
  while not pq.empty() :

    current_with_heuristic : a_star_pq_Type = pq.get(block=False)
    current : list[Coord] = current_with_heuristic[1]

    #print("A*")
    #print(index, isColumn)
    #print("\ncurrent cost   : " + str(len(current)))
    #print("heuristic cost : " + str(current_with_heuristic[PQ_TYPE_HEURISTIC_VALUE_INDEX])) 

    [derived_board, didElimLine] = deriveBoardForAStar(BOARD, current, PlaceColour)
    #print(render_board(derived_board, None, ansi=True)) # remove for submission

    lastPlaced : Coord = current[-1]
    for n in coordEmptySquareNeighbors(derived_board, lastPlaced) :
      neighbor : Coord = n

      next : list[Coord] = current + [neighbor]

      if neighbor not in pq.queue :

        if addToPQ_returnTrueIfSolution(next, derived_board) :
          return next


  return None











##################################################################################################################################################################################################
# COMPILE TIME
# COMPILE TIME
# COMPILE TIME
##################################################################################################################################################################################################











# a 9 * 0 matrix can include all pieces around the center 4,4

# could use board size but eh
matrix = [
       #
  "000000000",
  "000000000",
  "000011000",
  "000011000",
  "000000000", # 
  "000000000",
  "000000000",
  "000000000",
  "000000000",

]

#assert(matrix.__len__() == 9)
#assert(matrix.__len__() == matrix[0].__len__())


MID : int = int((len(matrix) - 1) / 2)
#assert(MID == 4)
CENTER = Coord(MID, MID)




def printMatrix(matrix) :

  for row in matrix:
    print(row)
  print("is equal to")

  print("PlaceAction(")

  count = 0
  for r, row in enumerate(matrix):
    for c, value in enumerate(row):
      if value == "1" :
        print(" Coord(" + str(r) +"," + str(c) +"),")
        count+=1 

  print(")")

  #assert(count == 4)

  
def coordsToPlaceAction(coords : list[Coord]) -> PlaceAction :
  #assert(len(coords) == 4)
  return PlaceAction(coords[0], coords[1], coords[2], coords[3])


def placeActionToOrderedCoords(place : PlaceAction) -> list[Coord] :
  return [place.c1, place.c2, place.c3, place.c4]



def offsetPlaceAction(place : PlaceAction, add : Coord, sub : Coord) -> PlaceAction :

  # do seperatly to avoid negatives
  coords = list(map(lambda c : c.__add__(add), place.coords))
  coords = list(map(lambda c : c.__sub__(sub), coords))
  return coordsToPlaceAction(coords)


"""
# passing

offset_actual = offsetPlaceAction(
    PlaceAction(
      Coord(4,6),
      Coord(3,4),
      Coord(3,5),
      Coord(3,6)
    ),
    Coord(1,4),
    Coord(4,4)
  )

offset_expected = PlaceAction(
    Coord(1,6),
    Coord(0,4),
    Coord(0,5),
    Coord(0,6)
  )

a = (offset_actual == offset_expected)
if not a :
  print(offset_actual, offset_expected)
assert(a)

"""



def movePlaceActionIndexToCoord(place : PlaceAction, index : int, coord : Coord) -> PlaceAction :
  other_coord = placeActionToOrderedCoords(place)[index]
  return offsetPlaceAction(place, coord, other_coord)


def adjacentTo(place : PlaceAction, to : Coord) -> bool :

  for coord in place.coords:
    if to in coordSquareNeighbors(coord): 
      return True

  """
  print()
  print("not adj")
  
  print(place, to, end = "")
  print(" no adj 2")
  print(render_board(deriveBoard({to : PlayerColor.BLUE}, [place])))
  """
  return False


"""
assert( not adjacentTo(
  PlaceAction(
    Coord(10,5),
    Coord(8,5),
    Coord(0,5),
    Coord(9,5)
  ), Coord(1,4))
)

assert( not adjacentTo(
  PlaceAction(
    Coord(0,0),
    Coord(1,0),
    Coord(2,0),
    Coord(3,0)
  ), Coord(4,1))
)

"""



def adjacentToColour(board : Board, place : PlaceAction, colour : PlayerColor) -> bool :
  # get all colour, remove coords in place

  exisingCoords : set[Coord] = set(getCoordsOfColour(board, colour)) - place.coords # don't want place coords

  for c in exisingCoords :
    if adjacentTo(place, c) :
      return True
  
  return False



PIECES_FOR_GENERATING : list[PlaceAction] = [ 

  # I piece
  PlaceAction(
    Coord(0,4),
    Coord(1,4),
    Coord(2,4),
    Coord(3,4),
  ),

  # L piece
  PlaceAction(
    Coord(1,4),
    Coord(1,5),
    Coord(2,4),
    Coord(3,4),
  ),


  # Z piece

  PlaceAction(
    Coord(1,5),
    Coord(2,4),
    Coord(2,5),
    Coord(3,4),
  ),


  # T piece
  PlaceAction(
    Coord(2,3),
    Coord(2,4),
    Coord(2,5),
    Coord(3,4),
  ),

  # O piece
  PlaceAction(
    Coord(2,4),
    Coord(2,5),
    Coord(3,4),
    Coord(3,5),
  )

]

"""
for place in PIECES_FOR_GENERATING :
  assert(adjacentTo(place, CENTER))
"""




def rotateVector90(coord : Vector2, around : Vector2) -> Vector2 :  # using vector to avoid out of bounds

  xCoordToAround = coord.c - around.c 
  yCoordToAround = around.r - coord.r
  return Vector2(around.r + xCoordToAround , around.c + yCoordToAround)

"""
assert(rotateVector90(Vector2(1,1), Vector2(1,1)) == Vector2(1,1))
assert(rotateVector90(Vector2(0,0), Vector2(1,1)) == Vector2(0,2))
assert(rotateVector90(Vector2(0,1), Vector2(1,1)) == Vector2(1,2))
"""


def coordToVec2(coord : Coord) -> Vector2 :
  return Vector2(coord.r, coord.c)


def Vec2ToCoord(vec : Vector2) -> Coord :
  return Coord(vec.r, vec.c)



def rotatePiece90(place : PlaceAction) -> PlaceAction : 

  # rotate around 90
  vectors : map[Vector2] =  map(coordToVec2 ,place.coords)
  rotatedVectors : map[Vector2]  = map(lambda c : rotateVector90(c, Vector2(0,0)), vectors)

  # and then move to the lowest coords
  # used to avoid out of bounds errors
  min_r = min(map(lambda v : v.r, rotatedVectors))
  min_c = min(map(lambda v : v.c, rotatedVectors))
  offset = Vector2(min_r, min_c) 
  rotatedAndMovedVectors : map[Vector2] = map(lambda vec : vec - offset, rotatedVectors)
  rotatedAndMovedCoords : list[Coord] = list(map(Vec2ToCoord, rotatedAndMovedVectors))

  return coordsToPlaceAction(rotatedAndMovedCoords)


def printPlaceAction(place : PlaceAction) :
  print("  PlaceAction(")

  for coord in place.coords :
    print("   Coord(" + str(coord.r) +"," + str(coord.c) +"),")

  print("  ),")

def allPlaceOptionsForPiecesAroundCenter(pieces : PlaceActionLst, PlaceColour : PlayerColor) -> PlaceActionLst :
  print("COMPILE TIME CODE")

  BOARD_WITH_CENTER = { CENTER : PlayerColor.BLUE } # visuliase the center

  place_options : set[frozenset[Coord]] = set() # used to avoid duplicates

  for coord in coordSquareNeighbors(CENTER) :

    for piece in pieces :

      for i in range(MAX_PIECE_SIZE) : # each of the piece of the place action

        for _ in range(4) : # covers all rotations, cache as this can be quite complex

          piece = rotatePiece90(piece) # i don't think the rotation point should matter as long as it avoids negative numbers
          new_piece : PlaceAction = movePlaceActionIndexToCoord(piece, i, coord)
          if (not CENTER in new_piece.coords) : # if not overlapping


            already_found = new_piece.coords in place_options
            # print("already_found " + str(already_found))

            if (not already_found) :

              assert(adjacentTo(new_piece, CENTER)) 

              visulising = False
              if visulising :

                print("GENERATING FOUND")
                print("new_piece " + str(new_piece))
                print(render_board(deriveBoard(BOARD_WITH_CENTER, [new_piece], PlaceColour)[0]))
              
              else : # generating

                printPlaceAction(new_piece)
                
              place_options.add(frozenset(new_piece.coords))


  ret : list[PlaceAction] = [ coordsToPlaceAction([crd for crd in st]) for st in place_options]
  assert(len(ret) != 0)
  return ret




















# adjacent to CENTER
GENERATED_PIECE_PLACEMENTS : list[PlaceAction] = [

  PlaceAction(
   Coord(3,3),
   Coord(3,4),
   Coord(3,5),
   Coord(3,6),
  ),
  PlaceAction(
   Coord(3,2),
   Coord(3,3),
   Coord(3,4),
   Coord(3,5),
  ),
  PlaceAction(
   Coord(3,1),
   Coord(3,2),
   Coord(3,3),
   Coord(3,4),
  ),
  PlaceAction(
   Coord(2,4),
   Coord(0,4),
   Coord(3,4),
   Coord(1,4),
  ),
  PlaceAction(
   Coord(3,7),
   Coord(3,4),
   Coord(3,5),
   Coord(3,6),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(3,3),
   Coord(3,4),
   Coord(3,5),
  ),
  PlaceAction(
   Coord(3,2),
   Coord(3,3),
   Coord(3,4),
   Coord(2,2),
  ),
  PlaceAction(
   Coord(2,3),
   Coord(2,4),
   Coord(3,4),
   Coord(2,2),
  ),
  PlaceAction(
   Coord(2,4),
   Coord(3,3),
   Coord(3,4),
   Coord(1,4),
  ),
  PlaceAction(
   Coord(2,3),
   Coord(3,3),
   Coord(3,4),
   Coord(3,5),
  ),
  PlaceAction(
   Coord(4,6),
   Coord(3,4),
   Coord(3,5),
   Coord(3,6),
  ),
  PlaceAction(
   Coord(2,5),
   Coord(3,4),
   Coord(3,5),
   Coord(1,5),
  ),
  PlaceAction(
   Coord(2,4),
   Coord(3,4),
   Coord(1,4),
   Coord(1,5),
  ),
  PlaceAction(
   Coord(2,4),
   Coord(3,4),
   Coord(3,5),
   Coord(3,6),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(3,3),
   Coord(3,4),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(2,4),
   Coord(3,3),
   Coord(3,4),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(2,3),
   Coord(2,4),
   Coord(3,4),
   Coord(3,5),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(4,6),
   Coord(3,4),
   Coord(3,5),
  ),
  PlaceAction(
   Coord(2,4),
   Coord(2,5),
   Coord(3,4),
   Coord(1,5),
  ),
  PlaceAction(
   Coord(2,3),
   Coord(3,3),
   Coord(3,4),
   Coord(2,2),
  ),
  PlaceAction(
   Coord(2,3),
   Coord(3,2),
   Coord(3,3),
   Coord(3,4),
  ),
  PlaceAction(
   Coord(2,3),
   Coord(3,3),
   Coord(3,4),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(2,3),
   Coord(2,4),
   Coord(3,4),
   Coord(1,4),
  ),
  PlaceAction(
   Coord(2,4),
   Coord(3,3),
   Coord(3,4),
   Coord(3,5),
  ),
  PlaceAction(
   Coord(2,3),
   Coord(2,4),
   Coord(2,5),
   Coord(3,4),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(2,5),
   Coord(3,4),
   Coord(3,5),
  ),
  PlaceAction(
   Coord(2,5),
   Coord(3,4),
   Coord(3,5),
   Coord(3,6),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(3,4),
   Coord(3,5),
   Coord(3,6),
  ),
  PlaceAction(
   Coord(2,4),
   Coord(2,5),
   Coord(3,4),
   Coord(1,4),
  ),
  PlaceAction(
   Coord(3,2),
   Coord(3,3),
   Coord(3,4),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(2,3),
   Coord(2,4),
   Coord(3,3),
   Coord(3,4),
  ),
  PlaceAction(
   Coord(2,4),
   Coord(2,5),
   Coord(3,4),
   Coord(3,5),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,4),
   Coord(5,5),
   Coord(5,6),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,4),
   Coord(5,5),
   Coord(5,2),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,4),
   Coord(5,1),
   Coord(5,2),
  ),
  PlaceAction(
   Coord(5,4),
   Coord(5,5),
   Coord(5,6),
   Coord(5,7),
  ),
  PlaceAction(
   Coord(7,4),
   Coord(5,4),
   Coord(8,4),
   Coord(6,4),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,4),
   Coord(5,5),
   Coord(6,5),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,4),
   Coord(4,2),
   Coord(5,2),
  ),
  PlaceAction(
   Coord(7,4),
   Coord(5,4),
   Coord(5,5),
   Coord(6,4),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,4),
   Coord(5,5),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(6,6),
   Coord(5,4),
   Coord(5,5),
   Coord(5,6),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(5,4),
   Coord(3,5),
   Coord(5,5),
  ),
  PlaceAction(
   Coord(6,6),
   Coord(5,4),
   Coord(6,4),
   Coord(6,5),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,4),
   Coord(6,4),
   Coord(5,2),
  ),
  PlaceAction(
   Coord(7,4),
   Coord(5,4),
   Coord(6,4),
   Coord(7,3),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,4),
   Coord(6,3),
   Coord(7,3),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,4),
   Coord(6,4),
   Coord(6,5),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(5,4),
   Coord(5,5),
   Coord(6,4),
  ),
  PlaceAction(
   Coord(6,6),
   Coord(5,4),
   Coord(5,5),
   Coord(6,5),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,4),
   Coord(4,2),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(5,4),
   Coord(6,3),
   Coord(6,4),
   Coord(7,3),
  ),
  PlaceAction(
   Coord(7,4),
   Coord(5,4),
   Coord(6,3),
   Coord(6,4),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,2),
   Coord(5,4),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,4),
   Coord(6,3),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,4),
   Coord(5,5),
   Coord(6,4),
  ),
  PlaceAction(
   Coord(7,4),
   Coord(5,4),
   Coord(6,4),
   Coord(6,5),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(5,4),
   Coord(5,5),
   Coord(6,5),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(5,4),
   Coord(5,6),
   Coord(5,5),
  ),
  PlaceAction(
   Coord(5,4),
   Coord(5,5),
   Coord(5,6),
   Coord(6,5),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,4),
   Coord(6,3),
   Coord(5,2),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,4),
   Coord(6,3),
   Coord(6,4),
  ),
  PlaceAction(
   Coord(5,4),
   Coord(5,5),
   Coord(6,4),
   Coord(6,5),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(6,3),
   Coord(3,3),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(2,3),
   Coord(5,3),
   Coord(3,3),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(4,0),
   Coord(4,1),
   Coord(4,2),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(2,3),
   Coord(3,3),
   Coord(1,3),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(6,3),
   Coord(7,3),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,2),
   Coord(3,3),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(3,1),
   Coord(4,1),
   Coord(4,2),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(3,1),
   Coord(3,2),
   Coord(3,3),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(2,3),
   Coord(3,3),
   Coord(4,2),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(2,3),
   Coord(2,4),
   Coord(3,3),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(4,1),
   Coord(4,2),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(6,2),
   Coord(6,3),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(4,3),
   Coord(6,2),
   Coord(4,2),
   Coord(5,2),
  ),
  PlaceAction(
   Coord(5,2),
   Coord(3,3),
   Coord(4,2),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(3,1),
   Coord(3,2),
   Coord(4,2),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(6,2),
   Coord(4,3),
   Coord(5,2),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,2),
   Coord(6,3),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(3,2),
   Coord(4,1),
   Coord(4,2),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(5,2),
   Coord(3,2),
   Coord(4,2),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(2,3),
   Coord(3,2),
   Coord(3,3),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(5,2),
   Coord(4,1),
   Coord(4,2),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(5,3),
   Coord(5,2),
   Coord(4,2),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(3,2),
   Coord(3,3),
   Coord(4,2),
   Coord(4,3),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(5,5),
   Coord(6,5),
   Coord(3,5),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(2,5),
   Coord(3,5),
   Coord(5,5),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(2,5),
   Coord(3,5),
   Coord(1,5),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(4,6),
   Coord(4,7),
   Coord(4,8),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(5,5),
   Coord(7,5),
   Coord(6,5),
  ),
  PlaceAction(
   Coord(4,6),
   Coord(4,5),
   Coord(5,5),
   Coord(6,5),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(5,5),
   Coord(3,5),
   Coord(3,6),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(4,6),
   Coord(4,7),
   Coord(5,7),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(4,6),
   Coord(2,6),
   Coord(3,6),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(5,5),
   Coord(5,6),
   Coord(5,7),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(2,5),
   Coord(2,6),
   Coord(3,5),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(5,5),
   Coord(6,4),
   Coord(6,5),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(4,6),
   Coord(4,7),
   Coord(3,5),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(4,6),
   Coord(3,6),
   Coord(5,5),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(4,6),
   Coord(5,6),
   Coord(5,7),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(2,6),
   Coord(3,5),
   Coord(3,6),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(5,5),
   Coord(5,6),
   Coord(6,5),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(4,6),
   Coord(5,6),
   Coord(3,6),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(4,6),
   Coord(4,7),
   Coord(3,6),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(4,6),
   Coord(3,5),
   Coord(5,5),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(4,6),
   Coord(4,7),
   Coord(5,6),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(2,5),
   Coord(3,5),
   Coord(3,6),
  ),
  PlaceAction(
   Coord(4,6),
   Coord(4,5),
   Coord(5,5),
   Coord(5,6),
  ),
  PlaceAction(
   Coord(4,5),
   Coord(4,6),
   Coord(3,5),
   Coord(3,6),
  ),

]



"""


tmp_derived_board = deriveBoard(
  {
    Coord(0,8) : PlayerColor.BLUE, 
    Coord(9,8) : PlayerColor.BLUE, 
    Coord(10,8) : PlayerColor.BLUE, 
  }, TEST1_SOLUTION
)

tmp_target = Coord(9,8)

print("counts")
print(
  emptyPlacesCount(tmp_derived_board, tmp_target.r, isColumn=False),
  emptyPlacesCount(tmp_derived_board, tmp_target.c, isColumn=True ),
)

print(render_board(tmp_derived_board, tmp_target))

print(tmp_derived_board.keys(), list(BOARD_ITER))

assert(isPieceDeleted(tmp_derived_board, tmp_target))


#######

for place in GENERATED_PIECE_PLACEMENTS :
  assert(adjacentTo(place, CENTER))

  
#######
  

assert(wrappingIndexDistance(0, 10) == 1)
assert(wrappingIndexDistance(5, 6) == 1)
assert(wrappingIndexDistance(3, 6) == 3)


"""












































































########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

# PART B













# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent



State = Board


MAX_DEPTH = 3
START_STATE = {} # empty board

DEBUG = False
C = 0.01 # from Upper Confidence Bound formula

# These two numbers should increase together
ITERATIONS = 50
EXPLORE_MIN = 1
BRANCHING_FACTOR = 5
VALIDATE = True


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


# which coords

def columnsAndRowsOccupied(coords : Iterable[Coord]) :

  rows = set()
  columns = set()

  for coord in coords :
    rows.add(coord.r)
    columns.add(coord.c)

  return (columns, rows)

def columnsAndRowsOccupied_WithColour(board : Board, player : PlayerColor) :

  coords = list(map(lambda x : x[0], filter(lambda x : x[1] == player, board.items())))
  return columnsAndRowsOccupied(coords)

# number


def mulColumnsAndRowsOccupied__WithColour(board : Board, player : PlayerColor) :

  occupied = columnsAndRowsOccupied_WithColour(board, player)

  # multi used to avoid concentration on a single axis and line elims
  return len(occupied[0]) * len(occupied[1])
     




def coordsInBoardOfPlayer(board : Board, player : Player) -> Iterable[Coord] :
   return map(lambda item : item[0], filter(lambda item : item[1] == player, board.items()))

def numberOfCoordsInColumnAndRows(coords : Iterable[Coord]) -> tuple[list, list] :
  # filter board by player coords
  # then the number is just len

  row_counts = [0]*BOARD_N
  col_counts = [0]*BOARD_N

  for coord in coords :
     row_counts[coord.r] += 1
     col_counts[coord.c] += 1

  return (col_counts, row_counts)
  


def subTop3numberOfCoordsInColumnAndRows(board : Board, player : Player) -> int :

  coords = coordsInBoardOfPlayer(board, player)
  (col_counts, row_counts) = numberOfCoordsInColumnAndRows(coords)
  allCounts = col_counts + row_counts
  return sum(sorted(allCounts, reverse=True)[:3])




def playerSquareBias(board : Board, player : Player) -> int :

  counts = Counter(board.values())
  return counts[player] - counts[reversePlayer(player)]
   

def heuristic(stateBeforeAction : State, action : Action, player : Player) -> float :
  # used to pick which value to expand
  # this is much better than expanding randomly

  heuristic_value = 0

  reversedPlayer = reversePlayer(player)
  stateAfterAction, isElim = deriveBoardWrapper(stateBeforeAction, [action], player)

  if isElim :
  
    heuristicSquareCountDifference = playerSquareBias(stateAfterAction, player)
    heuristic_value += heuristicSquareCountDifference * -1000 # if there is the possibility to eliminate, this should be important

  eliminationPrevention = - subTop3numberOfCoordsInColumnAndRows(stateAfterAction, player) + subTop3numberOfCoordsInColumnAndRows(stateAfterAction, reversedPlayer)
  heuristic_value += eliminationPrevention

  return heuristic_value



def isStateWin(state : State) -> Optional[Player] :
  if not PlayerColor.RED in state.values() : return PlayerColor.BLUE
  if not PlayerColor.BLUE in state.values() : return PlayerColor.RED
  return None
  
def tieBreaker(state : State) -> Optional[Player] :

  counts = Counter(state.values())

  if counts[PlayerColor.BLUE] > counts[PlayerColor.RED] : return PlayerColor.BLUE
  if counts[PlayerColor.RED] > counts[PlayerColor.BLUE] : return PlayerColor.RED
  return None



# TODO create iterator for memory maybe
allCoords = []
for x in range(BOARD_N) :
   for y in range(BOARD_N) :
      allCoords.append(Coord(x,y))


def getActionsFromState(state : State, PlaceColour : PlayerColor, isFirstMove : bool) -> Iterable[Action] :


  if not isFirstMove : 
    actions = placeActionsFromBoard(state, PlaceColour)
  
    return actions

  else :

      actions = set()

      for coord in allCoords:
            actions.update(coordPlaceOptions(state, coord))

            # there are a million options for the first move 
            # and you don't want to waste memory
            if len(actions) != 0 :
                break

      assert(len(actions) != 0)

      return actions





def applyActionToState(state : State, action : Action, PlaceColour : PlayerColor) -> State :
  return deriveBoardWrapper(state, [action], PlaceColour)[0]

def rolloutStrategy(state : State, player: Player) -> Optional[Action] :
  possibleActions = getActionsFromState(state, player, False)
  if possibleActions == [] :
     return None

  action = random.choice(list(possibleActions)) # radom rollout deals with running out of space
  return action









# ================================================================================
# impliment functions






# used to benchmark different implimentations
def deriveBoardWrapper(original_board : Board, placeActionLst : PlaceActionLst, PlaceColour : PlayerColor) -> tuple[Board, bool] :

  # call all functions once for cprofile

  result = deriveBoard(original_board, placeActionLst, PlaceColour)
  return result




class GameTree : # / node

  def __init__(self, children : list, winProp : WinsAndGames, action : Optional[Action]) -> None:
    self.children : list[GameTree] = children
    self.winProp : WinsAndGames = winProp # win proportion
    self.action = action 
    # state is derived, as it takes too much space


def printTree(tree : GameTree, state : Optional[State], playing : PlayerColor, toIndent = 100, indent = 0) :

  depth = indent + 1


  if indent > toIndent : return
  print("    "*indent, end ="")
  #print("action : " + str(tree.action), end ="")
  print(", winprop :" + str(tree.winProp), end ="")
  print("fract" + str(tree.winProp[0] / (tree.winProp[1] + 0.01))[1:4], end ="")
  #print(", state : " + str(state), end ="")
  print(" num children : " + str(len(tree.children)), end ="\n")

  for t in tree.children[:5] :
    tmpState = state
    if state != None and t.action :
      tmpState = applyActionToState(state, t.action, whosMoveFromDepth(depth=depth, playing=playing))
    printTree(t, state=tmpState, playing=playing, indent=(indent + 1), toIndent=toIndent)
  print()

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

def getMinMaxPath(tree : GameTree, isMaxFirst : bool, state : State, playing : PlayerColor) -> tuple[list[GameTree], State] :
  

  whosMove = playing

  path : list[GameTree] = [tree] # path indexes 
  while tree.children != [] :

    if (tree.action) :
      state = applyActionToState(state, tree.action, whosMove)

    next_i = getMinOrMaxFromChildren(tree, isMaxFirst)
    next = tree.children[next_i]
    path.append(next)
    tree = next
    isMaxFirst = not isMaxFirst 
    whosMove = reversePlayer(playing)

  if DEBUG :
    print("Path")
    for c in path :
      print(c.action)
    print("Path end")
    
  return path, state


def scoreFromTree(x : GameTree) :
  return scoreFromwinProp(x.winProp)

def rolloutSim(state : State, whosMove : Player, depth : int) -> Optional[Player] :

  while depth != MAX_DEPTH :
    break  # TODO

    action = rolloutStrategy(state, whosMove)
    if action == None :
       return reversePlayer(whosMove)

    state = applyActionToState(state, action, whosMove)
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

def makeMoveWith(initState : State, tree : GameTree, playerNotWhosMove: Player, isFirstMove : bool) -> GameTree :
  isMaxFirst = True

  # 1 Selection (min max)
  path, leafState = getMinMaxPath(tree, isMaxFirst, initState, playerNotWhosMove)
  depth = len(path)
  whosMoveNotPlayer = whosMoveFromDepth(depth=depth, playing=playerNotWhosMove)

  # 2 Expansion (add a single node)
  leafNode = path[-1]
  leafActions = list(getActionsFromState(leafState, whosMoveNotPlayer, isFirstMove))

  def heuristicFromAction(action : Action) :
    return heuristic(stateBeforeAction=leafState, action=action, player=whosMoveNotPlayer)

  leafActions.sort(key=heuristicFromAction, reverse=True)

  for action in leafActions[:BRANCHING_FACTOR] : 
    leafNode.children.append(GameTree([], (0, 0), action))
  
  if DEBUG :
    #print("Children actions ranked"); for t in leafNode.children: print(t.action)
    print()

  didWin = None
  if leafNode.children != [] :
     # there are no moves from this state

    path.append(leafNode.children[0])

    # 3 Simulation (rollout)

    # 3.1 derive state
    action = path[-1].action
    assert(action)
    state = applyActionToState(leafState, action, PlaceColour=whosMoveNotPlayer) 

    # 3.2 simluate rollout
    whoWon = rolloutSim(state, whosMoveNotPlayer, depth=depth)
    didWin = (whoWon == playerNotWhosMove)
  
  else :
    # other player won
    didWin = reversePlayer(whosMoveNotPlayer) == playerNotWhosMove

  # 4 Back-propagation (update win and games values)
  for curNode in path :
    assert(didWin != None)
    curNode.winProp = updateWinsAndGames(curNode.winProp, didWin)

  return tree

import math

def U(t : GameTree) -> float : return float(t.winProp[0])
def N(t : GameTree) -> float : return float(t.winProp[1])

def UCB(Parent_n : GameTree, n : GameTree) :
  assert(n in Parent_n.children)
  return (U(n) / N(n)) + C * math.sqrt(math.log(N(Parent_n), 2) / N(n))

def mcts(player : Player, fromState : State, isFirstMove : bool, iterations : int) -> Action :

  gameTree = GameTree([], (0,0), None) # starting node
  for _ in range(iterations) :
    gameTree = makeMoveWith(fromState, gameTree, player, isFirstMove)
    if DEBUG :
      print("Tree")
      printTree(gameTree, fromState, toIndent=3, playing=player)

  nodes, endState = getMinMaxPath(gameTree, True, fromState, player)
  bestAction = nodes[1].action # 1 to ignore start node
  assert(bestAction)
  return bestAction





def random_moves(player : Player, fromState : State, isFirstMove : bool) -> Action :

  actions = getActionsFromState(fromState, player, isFirstMove)
  return random.choice(list(actions))



def greedy_moves(player : Player, fromState : State, isFirstMove : bool) -> Action :

  def heuristicFromAction(action : Action) :
    return heuristic(stateBeforeAction=fromState, action=action, player=player)

  actions = list(getActionsFromState(fromState, player, isFirstMove))
  actions.sort(key=heuristicFromAction, reverse=True)

  return list(actions)[0] # can be an error here if no moves, this is fine




# ================================================================================
# call code



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



        # bruh
        def PlayerColorFromString(s) -> PlayerColor :
          if str(s) == "RED" : return PlayerColor.RED
          if str(s) == "BLUE" : return PlayerColor.BLUE
          assert(False)

        self._color : PlayerColor = PlayerColorFromString(color) 
        if color == PlayerColor.RED:
            print("Testing: I am playing as RED")
        if color == PlayerColor.BLUE:
            print("Testing: I am playing as BLUE")
        else :
           print("error on colour match") # TODO


        # init board state 
        self.board_state : Board = {}
        self.firstMove : bool = True # prob could use empty board, but could introduct weird bugs, this is easier
              


    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """

        # Below we have hardcoded two actions to be played depending on whether
        # the agent is playing as BLUE or RED. Obviously this won't work beyond
        # the initial moves of the game, so you should use some game playing
        # technique(s) to determine the best action to take.

        def get_action() :
           
          # if branching factor is too high, be quick and greedy

          if len(self.board_state) < 400000 :
            return greedy_moves(self._color, self.board_state, isFirstMove=self.firstMove)

          # if the branching factor is low, think ahead
          return mcts(self._color, self.board_state, iterations=ITERATIONS, isFirstMove=self.firstMove)


        IS_PROFILE = False

        if IS_PROFILE :


          import cProfile
          with cProfile.Profile() as pr:
              
              print("Print stats")
              
              action = get_action()
              
              pr.print_stats(sort='cumulative')
              print("end print stats")
        
        else : 
              
              action = get_action()


        # if a good move, you won't eliminate more of your own pieces
        if not (playerSquareBias(deriveBoard(self.board_state, [action], self._color)[0], self._color) >= playerSquareBias(self.board_state, self._color))  :
           
          print(render_board(self.board_state))
          print(render_board(deriveBoard(self.board_state, [action], self._color)[0]))
          print(self._color)
          assert(False)
           
        gc.collect() # reduce memory usage
        self.firstMove = False
        return action


    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after an agent has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        self.board_state = deriveBoardWrapper(self.board_state, [action], color)[0]
        







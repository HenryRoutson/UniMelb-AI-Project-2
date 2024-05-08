# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part A: Single Player Tetress

"""

I think for the next part they want use to use a strategy of trying to delete the line with the most pieces of the other colour

"""




from inspect import isclass
from itertools import groupby
from re import L
from .core import PlayerColor, Coord, PlaceAction, Direction, Vector2
from .utils import render_board
from typing import Callable, Optional
from typing import NewType
from queue import PriorityQueue
from collections import defaultdict
import math
import copy
import cProfile
import time

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


# impl

def search(
    board: Board, 
    target: Coord
) -> PlaceActionLst | None:
    """
    This is the entry point for your submission. You should modify this
    function to solve the search problem discussed in the Part A specification.
    See `core.py` for information on the types being used here.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.  
        `target`: the target BLUE coordinate to remove from the board.
    
    Returns:
        A list of "place actions" as PlaceAction instances, or `None` if no
        solution is possible.
    """


    # For testing A* -------------------------------------------------

    #print(a_star({}, [Coord(5,5)], 1, True))
    #print(a_star({}, [Coord(5,5)], 1, False))

    #print(a_star({}, [Coord(8,8)], 0, True))
    #print(a_star({}, [Coord(8,8)], 0, False))


    #print(a_star({Coord(2,3) : PlayerColor.BLUE, Coord(1,3) : PlayerColor.BLUE}, [Coord(2,2)], 5, True))

    """
    for row in BOARD_ITER :
      for col in BOARD_ITER :
        for index in BOARD_ITER :
          for b in [True, False] :
            #print("=====================================")
            #print("=====================================")
            #print(target, index, b)
            (a_star(board, [Coord(row, col)], index, b))
            #print("=====================================")
            #print("=====================================")
    print("DONE")
    """




    # For generating PIECES_FOR_GENERATING -------------------------------------------------
    # printMatrix(matrix)

    # For generating GENERATED_PIECE_PLACEMENTS -------------------------------------------------
      
    #allPlaceOptionsForPiecesAroundCenter(PIECES_FOR_GENERATING)

    # For actual search -------------------------------------------------
    return heuristic_search(board, target) # type: ignore

    """
    with cProfile.Profile() as pr:
        # ... do something ...
        heuristic_search(board, target)

        pr.print_stats(sort='cumulative')
    """








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


def fillColumnOrRow(index : int, isColumn : bool) -> set[Coord] :

  a : set[Coord] = set()
  for i in BOARD_ITER :
    if (isColumn) :
      a.add(Coord(r=i, c=index))
    else :
      a.add(Coord(r=index, c=i))

  return a

def checkAndRemoveColumnOrRowFilled(board : Board, index : int, isColumn : bool) -> Optional[Board] :

  line : set[Coord] = fillColumnOrRow(index, isColumn)
  lineRemainder : set[Coord] = line - set(board.keys())
  isLineFilled : bool = len(lineRemainder) == 0

  if (isLineFilled) :
    #print("LINE ELIMINATION")
    #print(render_board(board))
    #print("LINE ELIMINATION")
    for key in line :
      board.pop(key, None)

  if (isLineFilled) :
    return board

  return None


def boardEliminateFilledRowsOrColumns(board : Board) -> tuple[Board, bool] :

  didElim = False
  for b in [True, False] :
    for i in BOARD_ITER :
      
      cur = checkAndRemoveColumnOrRowFilled(board, i, b)
      if cur != None :
        board = cur
        didElim = True


  return (board, didElim)




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
  
  

def a_star_len(BOARD : Board, coords_from : list[Coord], index : int, isColumn : bool) -> int :
  tmp = a_star(BOARD, coords_from, index, isColumn)
  if tmp == None : return -1
  else : return len(tmp)


def minSquaresOfColorToIndexRowOrColumn(board : Board, target_index : int, isColumn) -> int :
  # manhattan distance

  squares_of_place_colour : list[Coord] = getCoordsOfColour(board, PLACE_COLOUR)

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


def minSquareCostForRowOrColumn(board : Board, target : Coord, isColumn : bool) -> int :
  if isColumn : index = target.c
  else        : index = target.r

  empty_places_on_line = emptyPlacesCount(board, index, isColumn)
  
  tmp = minSquaresOfColorToIndexRowOrColumn(board, index, isColumn)
  if (tmp == -1) : return -1
  squares_to_line = max(tmp - 1, 0) # -1 becuase if you have a square on the line that doesn't matter, it only needs to be next to the line
  min_num_squares = squares_to_line + empty_places_on_line

  return min_num_squares


def numSquaresToPieces(i : int) :
  return  math.ceil(i / 4) # 


def heuristic1(derived_board : Board, placeActionLst : PlaceActionLst, target : Target) -> Heuristic_value | None :

  ##################################################################################

  current_cost = placeActionLst.__len__()
  
  ##################################################################################

  # number of squares

  remaining_squares = min(
    minSquareCostForRowOrColumn(derived_board, target, isColumn=False),
    minSquareCostForRowOrColumn(derived_board, target, isColumn=True),
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

def isValidPiecePlace(place : PlaceAction, board : Board) -> bool :

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
  

def printBoardWithSquareAndPiece(coord : Coord, place : PlaceAction) :
  print(render_board(deriveBoard({coord : PlayerColor.BLUE}, [place])[0], coord))

def coordPlaceOptions(board : Board, around : Coord) -> PlaceActionLst:
  # all place actions around the around coord

  """
  if (not around in board) or board[around] != PLACE_COLOUR : 
    assert(False)
    return [] # placement may have been eliminated
  """

  # assert(board[around] == PLACE_COLOUR) # to connect other pieces, this needs to be the place colour

  options = []

  for placement_adj_to_center in GENERATED_PIECE_PLACEMENTS :
      
      #assert(not overLap(placement_adj_to_center, CENTER))
      #assert(adjacentTo(placement_adj_to_center, CENTER))

      # this is used to move placeAction from being adjacent to center, to being adjacent to the empty square
      placement_adj_to_around = offsetPlaceAction(placement_adj_to_center, around, CENTER) 
      #assert(not overLap(placement_adj_to_around, around))
      #assert(adjacentTo(placement_adj_to_around, around))

      if isValidPiecePlace(placement_adj_to_around, board):
        options.append(placement_adj_to_around)

  #assert(len(set(options)) == len(options))

  return options
  
  

def placeActionsFromBoard(board : Board) -> PlaceActionLst:
  # return a list of all pieces you can place connecting to a placeAction

  neighbors = set()

  for coord in getCoordsOfColour(board, PLACE_COLOUR) :
    for placeActionNeighbor in coordPlaceOptions(board, coord):

        #assert(adjacentTo(placeActionNeighbor, coord))
        neighbors.add(placeActionNeighbor)

  ret : PlaceActionLst = list(neighbors)
  return ret



def qcopy(board : Board) -> Board :
  newBoard : Board = dict()
  for item in board.items() :

    key : Coord = item[0]
    value : PlayerColor = item[1]

    newBoard[key] = value

  return newBoard

def deriveBoard(original_board : Board, PlaceActionLst : PlaceActionLst) -> tuple[Board, bool] :

  board = copy.deepcopy(original_board)
  #board = qcopy(original_board)
  
  for place in PlaceActionLst :
     for coord in place.coords :
        
        if(coord in board.keys()) :  # all spaces should be empty
          board = boardEliminateFilledRowsOrColumns(board)[0] # unless elim
          #assert(not coord in board.keys())

        board[coord] = PLACE_COLOUR

  # TODO
  # return board
        
  return boardEliminateFilledRowsOrColumns(board)



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

def heuristic_search(BOARD : Board, TARGET : Target) ->  PlaceActionLst | None :
  """
  Adapted from https:#en.wikipedia.org/wiki/A*_search_algorithm
  BOARD and TARGET are contant
  """

  pq : PriorityQueue[pq_Type] = PriorityQueue()

  def addToPQ_returnTrueIfSolution(placeActions : PlaceActionLst, initial_derived_board : Board) -> bool :
     # this derived board includes an extra piece

    neighbor = placeActions[-1]
    [derived_board_plus, didElimLine] =  deriveBoard(initial_derived_board, [neighbor]) # include neighbor
    

    """
    # more performant to leave out
    froze = frozenset(derived_board_plus.items())
    if (froze in reached) : return False # already reached
    reached.add(froze)
    """

    

    # shouldn't already be in pq
    if not placeActions in (map(lambda x : x[1].placeLst,pq.queue)):

      cost : Heuristic_value | None = heuristic1(derived_board_plus, placeActions, TARGET)
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
    if BOARD[coord] == PLACE_COLOUR :
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

    [derived_board, didElimLine] = deriveBoard(BOARD, current)

    #print(render_board(derived_board, TARGET, ansi=True)) # remove for submission


    for neighbor in placeActionsFromBoard(derived_board) :

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

def deriveBoardForAStar(board : Board, current : list[Coord]) :
  return deriveBoard(board, [coordToPlaceAction(coord) for coord in current])


def getIndex(coord : Coord, isColumn : bool) :
  if isColumn : return coord.c
  return coord.r

def coordListToHashable(list : list[Coord]) :
  return ' '.join([(str(x.r) + str(x.c)) for x in list])

def a_star(BOARD : Board, coords_from : list[Coord], index : int, isColumn : bool) ->  list[Coord] | None :
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

    [derived_board, didElimLine] = deriveBoardForAStar(BOARD, current)
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
  vectors : list[Vector2] =  list(map(coordToVec2 ,place.coords))
  rotatedVectors : list[Vector2]  = list(map(lambda c : rotateVector90(c, Vector2(0,0)), vectors))

  # and then move to the lowest coords
  # used to avoid out of bounds errors
  min_r = min(map(lambda v : v.r, rotatedVectors))
  min_c = min(map(lambda v : v.c, rotatedVectors))
  offset = Vector2(min_r, min_c) 
  rotatedAndMovedVectors : list[Vector2] = list(map(lambda vec : vec - offset, rotatedVectors))
  rotatedAndMovedCoords : list[Coord] = list(map(Vec2ToCoord, rotatedAndMovedVectors))

  return coordsToPlaceAction(rotatedAndMovedCoords)


def printPlaceAction(place : PlaceAction) :
  print("  PlaceAction(")

  for coord in place.coords :
    print("   Coord(" + str(coord.r) +"," + str(coord.c) +"),")

  print("  ),")

def allPlaceOptionsForPiecesAroundCenter(pieces : PlaceActionLst) -> PlaceActionLst :
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
                print(render_board(deriveBoard(BOARD_WITH_CENTER, [new_piece])[0]))
              
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
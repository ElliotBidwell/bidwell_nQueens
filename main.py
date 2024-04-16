import random
import math

echo = False


# Class contains a dictionary used to store every state that is expanded during a search. Used to manage this data in
# an efficient and encapsulated manner.
class ExpandedStates:

    def __init__(self):
        # Create a new empty dictionary and start the count at 0
        self.expanded = {}
        self.count = 0

    # Add a new state to the dictionary, using its string representation as the key and the object itself as the
    # value.
    def add_to_expanded(self, state):
        self.expanded[str(state)] = state
        self.count += 1

    # Clears the expanded state dictionary
    def clear_expanded(self):
        self.expanded.clear()
        self.count = 0

    # Searches the expanded state dictionary for the existence of the passed in state.
    def search_expanded(self, state):
        try:
            temp = self.expanded[str(state)]
            return True
        except:
            return False


# A global object that is used by many functions and classes to track which and how many states have been expanded,
# as well as modify that data.
expanded_states = ExpandedStates()


class NqueenState:

    # Constructor
    def __init__(self, parent=None):
        # Tracks the next empty column for the next queen to be placed in
        if parent is not None: # If this is not the starting state
            # Pick up at same column as previous state
            self.currentCol = parent.currentCol
        else:
            # Else, start at column 0
            self.currentCol = 0

        self.cost = 0 # The heuristic cost of the state, based on number of pairs of queens threatening each other
        self.board = [] # 2D list representing the chess board
        self.queens = [] # List of tuples representing the coordinates of each queen
        self.action = None # 2-tuple representing the coordinates of the queen placed to transition to this state
        self.successors = [] # List to contain references to each of this state's successors
        self.parent = parent # Reference to this state's parent state node

    # Returns a formatted string visualizing the state's board
    def __str__(self):
        board_string = ""
        # Loop through each row/column and append the symbol in each space to the output string
        for row in self.board:
            for space in row:
                board_string += f'{space}\t'
            # Skip a line after the row
            board_string += "\n"

        return board_string

    # Sets the state's board to the 2D list the gets passed in, then calls self.update_queens() to keep the list
    # of queen coordinates up to date.
    def set_board(self, new_board):
        self.board = [[tile for tile in row] for row in new_board]
        self.update_queens()

    # Searches the 2D list self.board for the existence of each occurrence, if any, of the char 'Q' and puts
    # the 2-tuple of list indices of each occurrence in self.queens.
    def update_queens(self):
        self.queens.clear()
        for row_index, row in enumerate(self.board):
            for col_index, tile in enumerate(row):
                if tile == 'Q':
                    self.queens.append((row_index, col_index))

    # Places a queen 'Q' in the row specified in the parameters, in the next empty column on the board.
    def place_queen(self, row, col=None):
        # Place the queen
        if col is None:
            self.board[row][self.currentCol] = 'Q'
            # Set the state's action to the coordinates of the new queen
            self.action = (row, self.currentCol)
            # Set next empty column to next column
            self.currentCol += 1

        else:
            self.board[row][col] = 'Q'

        self.update_queens()


    # Returns a formatted string visualizing the boards of each of the state's successor states
    def print_successors(self):
        output_string = ""

        # Array to contain each row of the eventual output string as its own separate string
        output_array = []

        # Add a new blank row to the output array for each row in a state's board
        # plus one extra to show the actions/transitions
        for row in self.board:
            output_array.append("")
        output_array.append("")

        # Iterate through successors
        for state in self.successors:
            # For each row in the output_array/board
            for i in range(len(output_array) - 1):
                # Append each element of the current row of the board to the current row of the output_array
                for symbol in state.board[i]:
                    output_array[i] += f'{symbol} '
                output_array[i] += f'\t'

            output_array[-1] += f'{state.action}\t\t'

        # Collapse the output array of strings into a single string
        for row in output_array:
            # Append the row followed by newline
            output_string += f'{row}\n'

        return output_string

# Returns a list of state objects from generating the input state's successors
# Will always return n successors unless the state is being expanded is one
# in which all the queens have been placed and there are no more empty columns
def nqueen_successors(state):

    # List to contain successors
    successors = []

    # Check whether all the queens haven't already been placed
    if state.currentCol < len(state.board[0]):

        # For each row in the board, generate a new successor
        for i in range(len(state.board)):
            successor = NqueenState(state)
            successor.set_board(state.board)
            successor.place_queen(i)
            nqueen_compute_weight(successor)
            successors.append(successor)

    return successors


# Sets the heuristic weight of the given state. Calculated by summing the number of queens threatening the most
# previously placed queen with the heuristic weight of the state's parent, which amounts to the total pairs of
# queens threatening each other on the board
def nqueen_compute_weight(state):
    score = 0
    echo_string = ""
    # queen_row, queen_col = state.action

    echo = False

    if echo:
        print(f'State:\n{state}Queen at {state.action}\nAc:\t\t\tUp:\t\t\tDn:')

    for queen_coords in state.queens:
        temp_queens = state.queens.copy()
        temp_queens.sort(key=lambda queen: queen_coords[1])
        queen_row, queen_col = queen_coords



        for other_queen in temp_queens[temp_queens.index(queen_coords):]:
            other_row, other_col = other_queen

            if echo:
                print(f'Cur: {queen_coords}\nOth: {other_queen}\nCart. Distances:\n'
                      f'R(Y): {queen_row - other_row if queen_row > other_row else other_row - queen_row}\n'
                      f'C(X): {queen_col - other_col}')

            if queen_coords != other_queen:
                if queen_row - other_row == queen_col - other_col:
                    score -= 1
                elif other_row - queen_row == queen_col - other_col:
                    score -= 1
                elif other_row == queen_row:
                    score -= 1

    # for i in range(1, queen_col + 1):
    #
    #     if state.board[queen_row][queen_col - i] == 'Q':
    #         score -= 1
    #
    #     if queen_row - i >= 0 and state.board[queen_row - i][queen_col - i] == 'Q':
    #         score -= 1
    #
    #     if queen_row + i < len(state.board) and state.board[queen_row + i][queen_col - i] == 'Q':
    #         score -= 1
    #
    #     if echo:
    #         echo_string += f'{queen_row, queen_col - i}: {state.board[queen_row][queen_col - i]}\t'
    #
    #     if queen_row - i >= 0:
    #         echo_string += f'{queen_row - i, queen_col - i}: {state.board[queen_row - i][queen_col - i]}\t'
    #     else:
    #         echo_string += f'         \t'
    #
    #     if queen_row + i < len(state.board):
    #         echo_string += f'{queen_row + i, queen_col - i}: {state.board[queen_row + i][queen_col - i]}\t'
    #     else:
    #         echo_string += f'         \t'
    #
    #     if echo:
    #         print(echo_string)
    #         echo_string = ""

    # if queen_col + 1 < len(state.board):
    #
    #     # Check upper right diagonal
    #     if queen_row - 1 >= 0 and state.board[queen_row - 1][queen_col + 1] == '-':
    #         score -= 1
    #         # state.board[queen_row - 1][queen_col + 1] = 'U'
    #
    #     # Check upper right diagonal
    #     if queen_row + 1 < len(state.board) and state.board[queen_row + 1][queen_col + 1] == '-':
    #         score -= 1
    #         # state.board[queen_row + 1][queen_col + 1] = 'D'
    #
    #     # Check directly right
    #     if state.board[queen_row][queen_col + 1] == '-':
    #         score -= 1
    #         # state.board[queen_row][queen_col + 1] = 'F'

    state.cost = score

    return score

def nqueen_start(n):

    random.seed()
    state = NqueenState(None)
    # state.set_board([[f'{i + (j * n)}' for i in range(n)] for j in range(n)])
    state.set_board([['-' for i in range(n)] for j in range(n)])
    # state.place_queen(random.randint(0, len(state.board) - 1))

    return state

def check_genetic():
    return -1

def nqueen_heuristic_search(start, successorFunc, searchType, checkGoal=None):

    outer_iterations = 0
    goal = None

    # Perform hill-climbing search
    if searchType == "hill-climbing":
        # 3 Problems
        # 1. Escaping local max/min and over-traversing on way back up
        # 2.
        # 3. Incorporating number of queens left to place into eval function

        # Begin at start state
        state = start

        # Loop indefinitely
        while True:
            # TODO True? Double-check
            # queens_left serves as simulated annealing scheduling variable
            queens_left = (len(state.board) - state.currentCol)

            if queens_left == 0 and state.cost >= 0:
                goal = state
                min_reached = True
                break

            if not expanded_states.search_expanded(state):
                state.successors = successorFunc(state)
                expanded_states.add_to_expanded(state)

            if echo:
                print(f'Iter: {outer_iterations} State:\n{state}\nSuccessors:\n')
                message = ""
                for i in range((state.cost) * -1):
                    message += '----'
                # print(f'(-{state.cost * -1}) {message}')

            if len(state.successors) > 0:

                costs = [successor.cost - queens_left for successor in state.successors]
                new_state = state.successors[costs.index(max(costs))]

                if not expanded_states.search_expanded(new_state):
                    state = new_state
                elif len(costs) > 1:
                    temp_successors = state.successors.copy()
                    temp_successors.pop(costs.index(max(costs)))

                    # temp_probs = [(float(state.cost - successor.cost)/float(queens_left * 2)) * float(not expanded_states.search_expanded(successor)) for successor in temp_successors]
                    temp_probs = [(float(successor.cost - state.cost) / (float(queens_left * 2)) + 0.1) * float(
                        not expanded_states.search_expanded(successor)) for successor in temp_successors]

                    try:
                        temp_probs = [prob/math.fsum(temp_probs) for prob in temp_probs]
                        random.seed()
                        successor_index = random.randint(1, 100)

                        for successor in temp_successors:
                            upper_bound = temp_probs[temp_successors.index(successor)] * 100
                            if successor_index <= upper_bound:
                                state = successor
                                break
                            else:
                                successor_index -= upper_bound

                    except ZeroDivisionError:
                        state = state.parent

                else:
                    state = state.parent

            else:
                state = state.parent

            outer_iterations += 1

    if searchType == "genetic" and checkGoal is not None:

        start.successors = successorFunc(start)
        population = start.successors.copy()
        population_cap = int(len(start.board) / 2 if len(start.board) > 2 else len(start.board))
        # population_cap = 8

        while True:

            for chromosome in population.copy():
                population.extend(chromosome.successors)

            if len(population) >= population_cap:
                print()

            population.sort(key=lambda chromosome: chromosome.cost)

            if echo:
                print(f'Initial Population:\n')
                for chromosome in population:
                    print(f'{chromosome}')

            if len(population) > population_cap:
                del population[population_cap + 1:]

            if echo:
                print(f'Culled Population:\n')
                for chromosome in population:
                    print(f'{chromosome}')

    if searchType == "PSO":
        None

    if echo:
        print(f'Iter: {outer_iterations}')

    expanded_states.clear_expanded()

    return goal, outer_iterations

def genetic_crossover(state_0, state_1):
    board_0 = rotate_board(state_0.board.copy())
    board_1 = rotate_board(state_1.board.copy())

    random.seed()
    cut_point_0 = random.randint(1, int(len(board_0)/2))

    # Catch ValueError for cases where the board is small enough
    try:
        cut_point_1 = random.randint(int(len(board_0)/2) + 1, len(board_0) - 1)
    except :
        cut_point_1 = int(len(board_0) / 2) + 1


    new_board_0 = board_0[:cut_point_0] + board_1[cut_point_0:cut_point_1] + board_0[cut_point_1:]
    new_board_1 = board_1[:cut_point_0] + board_0[cut_point_0:cut_point_1] + board_1[cut_point_1:]

    # new_state_0 = NqueenState((state_0, state_1))
    # new_state_1 = NqueenState((state_0, state_1))

    new_state_0 = NqueenState(None)
    new_state_1 = NqueenState(None)

    nqueen_compute_weight(new_state_0)
    nqueen_compute_weight(new_state_1)

    new_state_0.set_board(rotate_board(new_board_0, True))
    new_state_1.set_board(rotate_board(new_board_1, True))

    return new_state_0, new_state_1


#
def rotate_board(board, clockwise=False):

    rotation_count = 1 if clockwise else 3

    rotated = board.copy()

    for i in range(rotation_count):
        # Transpose the matrix
        board_length = len(board)
        transposed = [[rotated[i][j] for i in range(board_length)] for j in range(board_length)]
    
        # Reverse each row
        rotated = [row[::-1] for row in transposed]

    return rotated

def check_start(state):
    return [False in [space == '-' for space in row] for row in state.board].count(True) == 0


echo = True

# results = []
# for i in range(1):
#     start_state = nqueen_start(8)
#     goal_state = nqueen_heuristic_search(start_state, nqueen_successors, "hill-climbing")
#     results.append(float(goal_state[1]))
#     print(f'Goal (n = {4}):\n{goal_state[0]}cost: {goal_state[0].cost}\n{float(goal_state[1])} iterations\n')
# print(f'Iteration Counts: {results}\nAverage: {math.fsum(results)/float(len(results))}')

# results = []
# for i in range(1):
#     start_state = nqueen_start(4)
#     goal_state = nqueen_heuristic_search(start_state, nqueen_successors, "genetic", check_genetic)
#     results.append(float(goal_state[1]))
#     print(f'Goal (n = {4}):\n{goal_state[0]}cost: {goal_state[0].cost}\n{float(goal_state[1])} generations\n')
# print(f'Iteration Counts: {results}\nAverage: {math.fsum(results)/float(len(results))}')

# states = []
# transitions = [-3, 0, -2, -5]
# transitions = [0, 0, 2, 2]
# transitions = [2, 6, 3, 1, 4, 0, 5]
# transitions = [5, 6, 3, 4, 5, 6, 5]
# transitions = [1, 2, 3, 4, 5, 6, 7]
# state = nqueen_start(4)
# i = 0
#
# for transition in transitions:
#     state.successors = nqueen_successors(state)
#     state = state.successors[transition]
#     print(f'State {i}:\n{state}cost: {state.cost}\n')
#     i += 1


state_0 = nqueen_start(4)
state_0.successors = nqueen_successors(state_0)

state_1a = state_0.successors[0]
state_1a.successors = nqueen_successors(state_1a)

state_2a = state_1a.successors[0]
state_2a.successors = nqueen_successors(state_2a)

state_3a = state_2a.successors[0]
state_3a.successors = nqueen_successors(state_3a)

state_4a = state_3a.successors[0]
state_4a.successors = nqueen_successors(state_4a)

# state_5a = state_4a.successors[0]
# state_5a.successors = nqueen_successors(state_5a)


state_1b = state_0.successors[3]
state_1b.successors = nqueen_successors(state_1b)

state_2b = state_1b.successors[3]
state_2b.successors = nqueen_successors(state_2b)

state_3b = state_2b.successors[3]
state_3b.successors = nqueen_successors(state_3b)

state_4b = state_3b.successors[3]
state_4b.successors = nqueen_successors(state_4b)

print(f'Parents:\n{state_4a}\n{state_4b}')

child_0, child_1 = genetic_crossover(state_4a, state_4b)

print(f'Children:\n{child_0}\n{child_1}')
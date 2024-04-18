import random
import math
from utility import *

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
        if parent is not None and type(parent) is tuple: # If this is not the starting state
            # Pick up at same column as previous state
            self.currentCol = parent[0].currentCol, parent[0].currentCol
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
            col = len(self.queens)

        self.board[row][col] = 'Q'
        self.update_queens()

    # Returns a formatted string visualizing the boards of each of the state's successor states
    def print_successors(self):
        return self.print_states(self.successors)

    #
    def print_states(self, states):
        output_string = ""

        # Array to contain each row of the eventual output string as its own separate string
        output_array = []

        # Add a new blank row to the output array for each row in a state's board
        # plus one extra to show the actions/transitions
        for row in states[0].board:
            output_array.append("")
        output_array.append("")

        # Iterate through successors
        for state in states:
            # For each row in the output_array/board
            for i in range(len(output_array) - 1):
                # Append each element of the current row of the board to the current row of the output_array
                for symbol in state.board[i]:
                    output_array[i] += f'{symbol} '
                output_array[i] += f'\t'

            output_array[-1] += f'Cost: {state.cost}\t\t'

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
    if len(state.queens) < len(state.board):

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

    echo = False

    if echo:
        print(f'State:\n{state}\nAc:\t\t\tUp:\t\t\tDn:')

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


# Takes two state objects as parameters and returns a tuple of two states resulting from a two-point genetic
# crossover in which the two points are randomly chosen.
def genetic_crossover(state_0, state_1):
    board_0 = rotate_board(state_0.board.copy(), True)
    board_1 = rotate_board(state_1.board.copy(), True)

    random.seed()

    # cut_point = random.randint(1, len(board_0) - 1)

    cut_point_0 = random.randint(0, min(int(len(board_0)/2), len(state_0.queens), len(state_1.queens)))
    # cut_point = random.randint(0, min(int(len(board_0) / 2), len(state_0.queens), len(state_1.queens)))

    # Catch ValueError for cases where the board is small enough
    try:
        cut_point_1 = random.randint(cut_point_0, min(len(board_0), len(state_0.queens), len(state_1.queens)))
    except :
        cut_point_1 = cut_point_0

    # board_0_head, board_0_body, board_0_tail = (
    #     board_0[:cut_point_0],
    #     board_0[cut_point_0:cut_point_1],
    #     board_0[cut_point_1:])
    #
    # board_1_head, board_1_body, board_1_tail = (
    #     board_1[:cut_point_0],
    #     board_1[cut_point_0:cut_point_1],
    #     board_1[cut_point_1:])

    new_board_0 = board_1[:cut_point_0] + board_0[cut_point_0:cut_point_1] + board_1[cut_point_1:]
    new_board_1 = board_0[:cut_point_0] + board_1[cut_point_0:cut_point_1] + board_0[cut_point_1:]

    # new_board_0 = board_1_head + board_0_body + board_1_tail
    # new_board_1 = board_0_head + board_1_body + board_0_tail

    # new_board_0 = board_1[:cut_point] + board_0[cut_point:]
    # new_board_1 = board_0[:cut_point] + board_1[cut_point:]

    new_state_0 = NqueenState((state_0, state_1))
    new_state_1 = NqueenState((state_0, state_1))

    # new_state_0 = NqueenState(None)
    # new_state_1 = NqueenState(None)

    nqueen_compute_weight(new_state_0)
    nqueen_compute_weight(new_state_1)

    new_state_0.set_board(rotate_board(new_board_0))
    new_state_1.set_board(rotate_board(new_board_1))

    nqueen_compute_weight(new_state_0)
    nqueen_compute_weight(new_state_1)

    for i, column in enumerate(rotate_board(state_0.board)):
        if column.count('Q') > 1:
            x = 0
        if rotate_board(state_1.board)[i].count('Q') > 1:
            x = 0

    return new_state_0, new_state_1


def nqueen_start(n):

    random.seed()
    state = NqueenState(None)
    # state.set_board([[f'{i + (j * n)}' for i in range(n)] for j in range(n)])
    state.set_board([['-' for i in range(n)] for j in range(n)])
    # state.place_queen(random.randint(0, len(state.board) - 1))

    return state


def mutate_population(population, pop_best):

    for mutation_index in range(len(population) - 1):
        random.seed()

        new_mutated_state = NqueenState()
        new_mutated_state.set_board(population[mutation_index].board)

        # if new_mutated_state.cost - (len(new_mutated_state.board) - len(new_mutated_state.queens)) < pop_best:

        for i in range(len(new_mutated_state.board) - 1):

            # Randomly choose the index of a queen to move to another random space in the same column
            try:
                mutation_result_index = random.randint(0, len(new_mutated_state.queens) - 1)
            except ValueError:
                mutation_result_index = len(new_mutated_state.queens) - 1

            # Get the tuple of coordinates corresponding to that queen
            mutated_queen_0 = new_mutated_state.queens[mutation_result_index]
            test = mutated_queen_0[0], mutated_queen_0[1]
            # Clear the space occupied by the queen
            new_mutated_state.board[mutated_queen_0[0]][mutated_queen_0[1]] = '-'
            # Choose a random row to move it to
            rand_row = random.randint(0, len(new_mutated_state.board) - 1)
            # Place the queen in the new row
            new_mutated_state.place_queen(rand_row, mutated_queen_0[1])

            if f'{new_mutated_state}' not in [f'{chromosome}' for chromosome in population]:
                population[mutation_index].set_board(new_mutated_state.board)
                break
            else:
                None


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
            queens_left = (len(state.board) - len(state.queens))

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

        population = [start]
        population_cap = int(len(start.board) / 2) + 1 if len(start.board) > 2 else len(start.board)

        mutation_increment = 1
        mutation_timer = 0
        best_rec_weight = -1 * len(start.board)

        # Loop until goal state found
        while goal is None:

            temp_population = population.copy()

            # Loop through entire population
            for chromosome in temp_population:

                # Create a copy of the population
                other_chromosomes = temp_population.copy()

                # Remove all occurrences of the current state (chromosome) from the population copy
                for other_chromosome in temp_population:
                    if check_boards_equal(chromosome, other_chromosome):
                        other_chromosomes.remove(other_chromosome)

                while len(other_chromosomes) > 0:

                    random.seed()
                    random_partner_index = random.randint(0, len(other_chromosomes) - 1)
                    random_partner = other_chromosomes.pop(random_partner_index)
                    child_0, child_1 = genetic_crossover(chromosome, random_partner)
                    outer_iterations += 1

                    if f'{child_0}' not in [f'{chromosome}' for chromosome in population]:
                        population.append(child_0)

                    if f'{child_1}' not in [f'{chromosome}' for chromosome in population]:
                        population.append(child_1)

            population.sort(key=lambda chromosome: chromosome.cost - (len(chromosome.board) - len(chromosome.queens)),
                            reverse=True)

            other_chromosomes = population.copy()

            for chromosome in other_chromosomes:
                queens_left = len(chromosome.board) - len(chromosome.queens)

                # Check for goal state, set return value to current state and break loop if true
                if chromosome.cost == 0 and queens_left == 0:
                    goal = chromosome
                    break

                # If the state hasn't been expanded, expand its successors and mark it as expanded
                if not expanded_states.search_expanded(chromosome):
                    chromosome.successors = successorFunc(chromosome)
                    expanded_states.add_to_expanded(chromosome)

                # Add state's successors to population
                for successor in chromosome.successors:
                    if f'{successor}' not in [f'{chromosome}' for chromosome in population]:
                        population.append(successor)
                    else:
                        x = 0

                population.sort(
                    key=lambda chromosome: chromosome.cost - (len(chromosome.board) - len(chromosome.queens)),
                    reverse=True)

                if len(population) > population_cap:
                    population = population[:population_cap]

                best_pop_weight = population[0].cost - (len(population[0].board) - len(population[0].queens))

                if mutation_timer >= mutation_increment:
                    # Reset the mutation timer
                    mutation_timer = 0

                    if best_pop_weight > best_rec_weight + 1:
                        best_rec_weight = best_pop_weight

                    else:
                        mutate_population(population, best_pop_weight)

            mutation_timer += 1

    # TODO Implement PSO
    if searchType == "PSO":
        None

    if echo:
        print(f'Iter: {outer_iterations}')

    expanded_states.clear_expanded()

    return goal, outer_iterations


echo = False

results = []
tests = 100
n = 6
test_types = ["hill-climbing", "genetic", "PSO"]
test_type = 1
for i in range(tests):
    start_state = nqueen_start(n)
    goal_state = nqueen_heuristic_search(start_state, nqueen_successors, test_types[test_type], check_genetic)
    results.append(float(goal_state[1]))
    print(f'Test {i + 1}/{tests} (n = {n}) Goal:\n{goal_state[0]}cost: {goal_state[0].cost}\n{float(goal_state[1])} iterations/generations\n')
print(f'Test Type: {test_types[test_type]}\nn = {n}\n# of tests: {tests}\nIteration Counts: {results}\nAverage: {math.fsum(results)/float(len(results))}')

# Results:
#
# Test Type: genetic
# n = 6
# # of tests: 5
# Iteration Counts: [376.0, 68.0, 1157.0, 672.0, 860.0]
# Average: 626.6

# Test Type: hill-climbing
# n = 6
# # of tests: 5
# Iteration Counts: [28441.0, 14018.0, 19099.0, 19093.0, 19103.0]
# Average: 19950.8
#
# Test Type: genetic (AFTER)
# n = 6
# # of tests: 5
# Iteration Counts: [71727.0, 252837.0, 41031.0, 276273.0, 178133.0]
# Average: 164000.2











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


# state_0 = nqueen_start(5)
# state_0.successors = nqueen_successors(state_0)
#
# state_1a = state_0.successors[0]
# state_1a.successors = nqueen_successors(state_1a)
#
# state_2a = state_1a.successors[0]
# state_2a.successors = nqueen_successors(state_2a)
#
# state_3a = state_2a.successors[0]
# state_3a.successors = nqueen_successors(state_3a)
#
# state_4a = state_3a.successors[0]
# state_4a.successors = nqueen_successors(state_4a)
#
# state_5a = state_4a.successors[0]
# state_5a.successors = nqueen_successors(state_5a)
#
#
# state_1b = state_0.successors[4]
# state_1b.successors = nqueen_successors(state_1b)
#
# state_2b = state_1b.successors[4]
# state_2b.successors = nqueen_successors(state_2b)
#
# state_3b = state_2b.successors[4]
# state_3b.successors = nqueen_successors(state_3b)
#
# state_4b = state_3b.successors[4]
# state_4b.successors = nqueen_successors(state_4b)
#
# state_5b = state_4b.successors[4]
# state_5b.successors = nqueen_successors(state_5b)

# print(f'Parents:\n{state_4a}\n{state_4b}')
#
# child_0, child_1 = genetic_crossover(state_5a, state_5b)
#
# print(f'Children:\n{child_0}\n{child_1}')
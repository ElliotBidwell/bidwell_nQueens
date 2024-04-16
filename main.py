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
    def __init__(self, parent):
        # Tracks the current column for the next piece to be placed in
        if parent is not None:
            self.currentCol = parent.currentCol
        else:
            self.currentCol = 0
        # The total cost to traverse the tree along the path to this node
        self.cost = 0
        # List representing the puzzle grid to contain lists to represent rows of tiles and their contents
        self.board = []
        # Reserved for a 2-tuple representing the direction vector the blank space move in to transition to this state
        self.action = None
        # List to contain references to each of this state's successors that are generated
        self.successors = []
        # Reference to this state's parent state node
        self.parent = parent

    def __str__(self):
        board_string = ""
        # Loop through each row/column and append the number in each space to the output string
        for row in self.board:
            for space in row:
                board_string += f'{space}\t'
            # Skip a line after the row
            board_string += "\n"

        return board_string

    # Sets the state's board to the 2D list the gets passed in.
    def set_board(self, new_board):
        self.board = [[tile for tile in row] for row in new_board]

    def place_queen(self, row):
        self.board[row][self.currentCol] = 'Q'
        self.action = (row, self.currentCol)
        self.currentCol += 1

    def print_successors(self):
        output_string = ""

        # Array to contain each row of the eventual output string as its own separate string
        output_array = []

        # Add a new blank row to the output array for each row in a single states board
        # plus one extra to show the actions/transitions
        for row in self.board:
            output_array.append("")
        output_array.append("")

        # Iterate through states in the sequence
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

def nqueen_compute_weight(state):
    score = 0
    queen_y, queen_x = state.action
    echo_string = ""

    echo = False

    if echo:
        print(f'State:\n{state}Queen at {state.action}\nAc:\t\t\tUp:\t\t\tDn:')
    for i in range(1, queen_x + 1):

        if state.board[queen_y][queen_x - i] == 'Q':
            score -= 1

        if queen_y - i >= 0 and state.board[queen_y - i][queen_x - i] == 'Q':
            score -= 1

        if queen_y + i < len(state.board) and state.board[queen_y + i][queen_x - i] == 'Q':
            score -= 1

        if echo:
            echo_string += f'{queen_y, queen_x - i}: {state.board[queen_y][queen_x - i]}\t'

        if queen_y - i >= 0:
            echo_string += f'{queen_y - i, queen_x - i}: {state.board[queen_y - i][queen_x - i]}\t'
        else:
            echo_string += f'         \t'

        if queen_y + i < len(state.board):
            echo_string += f'{queen_y + i, queen_x - i}: {state.board[queen_y + i][queen_x - i]}\t'
        else:
            echo_string += f'         \t'

        if echo:
            print(echo_string)
            echo_string = ""

    # if queen_x + 1 < len(state.board):
    #
    #     # Check upper right diagonal
    #     if queen_y - 1 >= 0 and state.board[queen_y - 1][queen_x + 1] == '-':
    #         score -= 1
    #         # state.board[queen_y - 1][queen_x + 1] = 'U'
    #
    #     # Check upper right diagonal
    #     if queen_y + 1 < len(state.board) and state.board[queen_y + 1][queen_x + 1] == '-':
    #         score -= 1
    #         # state.board[queen_y + 1][queen_x + 1] = 'D'
    #
    #     # Check directly right
    #     if state.board[queen_y][queen_x + 1] == '-':
    #         score -= 1
    #         # state.board[queen_y][queen_x + 1] = 'F'

    state.cost = state.parent.cost + score

    return score

def nqueen_random_start(n):

    random.seed()
    state = NqueenState(None)
    # state.set_board([[f'{i + (j * n)}' for i in range(n)] for j in range(n)])
    state.set_board([['-' for i in range(n)] for j in range(n)])
    # state.place_queen(random.randint(0, len(state.board) - 1))

    return state

def nqueen_heuristic_search(start, checkGoal, successorFunc, searchType):

    outer_iterations = 0
    goal = None

    # Perform hill-climbing search
    if searchType == "hill-climbing":
        # Begin at start state
        state = start
        # state.cost = -31
        min_reached = False

        alt = True

        # Loop indefinitely
        while True:
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

            if check_start(state):
                print("FINALLY")

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


                    # temp = [expanded_states.search_expanded(successor) for successor in state.successors]
                    #
                    # try:
                    #     temp_state = state.successors[temp.index(False)]
                    #
                    #     if temp_state.cost - state.cost <= 0 or state.parent is None:
                    #         state = state.successors[temp.index(False)]
                    #     else:
                    #         print("Huh?")
                    #
                    # except ValueError:
                    #     state = state.parent

                else:
                    state = state.parent

            else:
                state = state.parent

            outer_iterations += 1

    if echo:
        print(f'Iter: {outer_iterations}')

    expanded_states.clear_expanded()

    return goal, outer_iterations


def check_start(state):
    return [False in [space == '-' for space in row] for row in state.board].count(True) == 0


echo = False
# start_state = nqueen_random_start(7)
# print(check_start(start_state))
# goal_state = nqueen_heuristic_search(start_state, 1, nqueen_successors, "hill-climbing")
# print(f'Start:\n{start_state}cost: {start_state.cost}\n')
# print(f'Goal:\n{goal_state}cost: {goal_state.cost}\n')

results = []
for i in range(10):
    start_state = nqueen_random_start(8)
    goal_state = nqueen_heuristic_search(start_state, 1, nqueen_successors, "hill-climbing")
    results.append(float(goal_state[1]))
    # print(f'Goal (n = {7}):\n{goal_state[0]}cost: {goal_state[0].cost}\n')
print(f'Iteration Counts: {results}\nAverage: {math.fsum(results)/float(len(results))}')

# state = start_state
# for i in range(10):
#     for successor in
#     state = state.successors[i]
#     print(f'State {i}:\n{state}cost: {state.cost}\n')
#     state.successors = nqueen_successors(state)
#     i += 1

# states = []
# transitions = [-3, 0, -2, -5]
# print(max(transitions))
# # transitions = [2, 6, 3, 1, 4, 0, 5]
# # transitions = [5, 6, 3, 4, 5, 6, 5]
# # transitions = [1, 2, 3, 4, 5, 6, 7]
# state = start_state
# i = 0
#
# for transition in transitions:
#     state = state.successors[transition]
#     print(f'State {i}:\n{state}cost: {state.cost}\n')
#     state.successors = nqueen_successors(state)
#     i += 1



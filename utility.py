import random

def rotate_board(board, clockwise=False):
    """
    This function rotates a 2D list to exchange its columns with rows and vise versa. By default, it rotates
    counter-clockwise, otherwise it rotates 3 times to emulate a clockwise rotation. It is used by the genetic algorithm
    so that the columns could be used to represent a sequence of genes that can be sliced and grafted to other
    chromosomes.

    :param board: The board to be rotated
    :param clockwise: Whether the rotation is clockwise
    :return rotated_board: The resulting board
    """

    rotation_count = 1 if clockwise else 3

    for j in range(len(board[0])):
        curr = [column[j] for column in board]
        prev = [column[j - 1] for column in board]
        if curr.count('Q') >= 1 and prev.count('Q') == 0 and j > 0:
            x = 0

    rotated_board = board.copy()

    for i in range(rotation_count):
        # Transpose the matrix
        board_length = len(board)
        transposed = [[rotated_board[i][j] for i in range(board_length)] for j in range(board_length)]

        # Reverse each row
        rotated_board = [row[::-1] for row in transposed]

    return rotated_board


def select_outcome(probs):
    """
    Selects an outcome based on the given list of each of their probabilities.

    :param probs: A list of floats list representing probabilities summing to 1.0
    :return: The index of the selected outcome.
    """

    if sum(probs) == 1.0:
        cumulative_probabilities = [sum(probs[:i + 1]) for i in range(len(probs))]
        random_value = random.random()

        for i, cumulative_probability in enumerate(cumulative_probabilities):
            if random_value < cumulative_probability:
                return i
    else:
        return -1


def check_start(state):
    return [False in [space == '-' for space in row] for row in state.board].count(True) == 0


def check_boards_equal(state_0, state_1):
    return True not in [False in [state_0.board[i][j] == state_1.board[i][j] for j in range(len(state_0.board))] for i in range(len(state_0.board))]


def check_genetic():
    return -1
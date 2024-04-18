import random

#
def rotate_board(board, clockwise=False):
    rotation_count = 1 if clockwise else 3

    for j in range(len(board[0])):
        curr = [column[j] for column in board]
        prev = [column[j - 1] for column in board]
        if curr.count('Q') >= 1 and prev.count('Q') == 0 and j > 0:
            x = 0

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


def check_boards_equal(state_0, state_1):
    return True not in [False in [state_0.board[i][j] == state_1.board[i][j] for j in range(len(state_0.board))] for i in range(len(state_0.board))]


def check_genetic():
    return -1
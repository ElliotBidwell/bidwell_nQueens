import random
import math
import time
import statistics as stats
from utility import *
import numpy as np


#
class ExpandedStates:
    """
    Class contains a dictionary used to store every state that is expanded during a search, as well as the number of
    states in the dictionary. It is used to manage this data in an efficient and encapsulated manner.
    """

    def __init__(self):
        """
        Create a new empty dictionary and start the count at 0
        """
        self.expanded = {}
        self.count = 0

    def add_to_expanded(self, state):
        """
        Add the passed in state to the dictionary and update the count, using its string representation as the key
        and the object itself as the value.

        :param state: The state to be added
        """

        self.expanded[str(state)] = state
        self.count += 1

    def clear_expanded(self):
        """
        Clears the expanded state dictionary and resets the count.
        """

        self.expanded.clear()
        self.count = 0

    def search_expanded(self, state):
        """
        Searches the expanded state dictionary for the existence of the passed in state and returns a boolean
        indicating whether the state was found.

        :param state:
        :return: True if found, else false
        """

        try:
            temp = self.expanded[str(state)]
            return True
        except:
            return False


class ComplexityTracker:
    """
    This class defines an object that keeps track of various performance metrics gathered as a search executes.
    It tracks the number of inner/outer loop iterations, mutations, crossovers, and successors generated.
    """

    def __init__(self):
        self.inner_iterations = 0
        self.outer_iterations = 0
        self.mutations = 0
        self.crossovers = 0
        self.successors_generated = 0

    """
    The following are functions to either increment one of the counters by 1 or reset them back to 0
    """
    def increment_inner(self):
        self.inner_iterations += 1

    def increment_outer(self):
        self.outer_iterations += 1

    def increment_mutations(self):
        self.outer_iterations += 1

    def increment_crossovers(self):
        self.outer_iterations += 1

    def increment_successors(self):
        self.outer_iterations += 1

    def reset_inner(self):
        self.inner_iterations = 0

    def reset_outer(self):
        self.outer_iterations = 0

    def reset_mutations(self):
        self.outer_iterations += 1

    def reset_crossovers(self):
        self.outer_iterations += 1

    def reset_successors(self):
        self.outer_iterations += 1

    def reset_all_counters(self):
        """
        This functions resets all counters
        """

        self.reset_inner()
        self.reset_outer()
        self.reset_mutations()
        self.reset_crossovers()
        self.reset_successors()


class NqueenState:
    """
    This class defines an object representing a single state in the state space. It contains class variables for
    several pieces of data used by the search algorithms during their execution. This includes the state's board,
    cost, queen locations, successors, velocity, and personal best fitness/cost.
    """

    # Constructor
    def __init__(self, parent=None, board=[]):
        # Tracks the next empty column for the next queen to be placed in
        if parent is not None and type(parent) is tuple: # If this is not the starting state
            # Pick up at same column as previous state
            self.currentCol = parent[0].currentCol, parent[0].currentCol
        else:
            # Else, start at column 0
            self.currentCol = 0

        # The heuristic cost of the state, based on number of pairs of queens threatening each other
        self.cost = 0
        # 2D list representing the chess board
        self.board = board
        # List of tuples representing the coordinates of each queen
        self.queens = []
        # 2-tuple representing the coordinates of the queen placed to transition to this state
        self.action = None
        # List to contain references to each of this state's successors
        self.successors = []
        # Reference to this state's parent state node
        self.parent = parent
        # A 2D list of floats with the same dimensions as the board where each value is the velocity component
        # that corresponds to the matching space on the board
        self.velocity = []
        # The best cost/fitness that this state has achieved
        self.personal_best = float("-inf")

    def __str__(self):
        """
        Returns a formatted string visualizing the state's board

        :return board_string: String representation of the board
        """
        board_string = ""
        # Loop through each row/column and append the symbol in each space to the output string
        for row in self.board:
            for space in row:
                board_string += f'{space}\t'
            # Skip a line after the row
            board_string += "\n"

        return board_string

    def set_board(self, new_board):
        """
        Sets the state's board to the 2D list that gets passed in, then calls self.update_queens() to keep the list
        of queen coordinates up to date.

        :param new_board:
        :return:
        """

        self.board = [[tile for tile in row] for row in new_board]
        self.update_queens()

    def set_velocity(self, new_velocity=None):
        """
        Sets the state's velocity to the 2D list that gets passed in.

        :param new_velocity: The velocity list to replace the existing one.
        """

        if new_velocity is None:
            self.velocity = [[0 for _ in range(len(self.board))] for _ in range(len(self.board))]
        else:
            self.velocity = new_velocity.copy()

    def set_cost(self, new_cost):
        """
        Sets the cost of the state to the value passed in. If the new value is greater than the personal best, then
        the personal best is set to that value

        :param new_cost: The new cost
        """

        if new_cost > self.personal_best:
            self.personal_best = new_cost

        self.cost = new_cost

    def update_queens(self):
        """
        Updates the list of queen coordinates Searches the 2D list self.board for the existence of each occurrence,
        if any, of the char 'Q' and puts the 2-tuple of list indices of each occurrence in self.queens.

        :return:
        """

        self.queens.clear()
        for row_index, row in enumerate(self.board):
            for col_index, tile in enumerate(row):
                if tile == 'Q':
                    self.queens.append((row_index, col_index))

    def place_queen(self, row, col=None):
        """
        Places a queen 'Q' in the row specified in the parameters, or in the next empty column on the board if no
        column is given.

        :param row:
        :param col:
        :return:
        """

        if col is None:
            col = len(self.queens)

        self.board[row][col] = 'Q'
        self.update_queens()

    def remove_queen(self, col):
        """
        Replaces the queen in the specified column with a blank space. It does so be iterating through the list of queen
        coordinates and replacing the queen whose column matches the one specified.

        :param col:
        :return:
        """
        for queen in self.queens:
            if queen[1] == col:
                self.board[queen[0]][col] = '-'
                break
        self.update_queens()

    def print_successors(self):
        """
        Calls print_states() to return a formatted string visualizing the boards of each of the state's successor states

        :return: Formatted successor string
        """
        return self.print_states(self.successors)

    def print_states(self, states):
        """
        Returns a formatted string made up of the string representations of each state in the list that is passed in

        :param states: The list of states
        :return output_string: Formatted string representation of states
        """

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


# Globals for toggling echo mode and verbose echo (echo_v) mode
echo = False
echo_v = False

# TODO
complexTracker = ComplexityTracker()

# A global object that is used by many functions and classes to track which and how many states have been expanded,
# as well as modify that data.
expanded_states = ExpandedStates()

# 
def nqueen_successors(state):
    """
    Returns a list of NqueenState objects from generating the inputted state's successors.
    It will always return n successors unless the state is being expanded is one
    in which all the queens have been placed and there are no more empty columns

    :param state: The NqueenState whose successors are to be generated
    :return successors: A list of the state's successors
    """

    # List to contain successors
    successors = []

    # Check whether all the queens haven't already been placed
    if len(state.queens) < len(state.board):

        # For each row in the board, generate a new successor
        for i in range(len(state.board)):
            complexTracker.increment_inner()

            successor = NqueenState(state)
            successor.set_board(state.board)
            successor.place_queen(i)
            nqueen_compute_weight(successor)
            successors.append(successor)

    return successors


def nqueen_compute_weight(state):
    """
    Sets the heuristic weight of the given state. Calculated as the total pairs of
    queens threatening each other on the board.

    :param state: The NqueenState whose cost is to be set
    :return score: The determined cost
    """
    score = 0

    for queen_coords in state.queens:

        complexTracker.increment_outer()

        temp_queens = state.queens.copy()
        temp_queens.sort(key=lambda queen: queen_coords[1])
        queen_row, queen_col = queen_coords

        for other_queen in temp_queens[temp_queens.index(queen_coords):]:
            complexTracker.increment_inner()

            other_row, other_col = other_queen

            if queen_col != other_col:
                if queen_row - other_row == queen_col - other_col:
                    score -= 1
                if other_row - queen_row == queen_col - other_col:
                    score -= 1
                if other_row == queen_row:
                    score -= 1

    state.cost = score

    return score


def genetic_crossover(state_0, state_1):
    """
    Takes two state objects as parameters and returns a tuple of two states resulting from a two-point genetic
    crossover in which the two points are randomly chosen.

    :param state_0: One of the two NqueenStates about to undergo crossover with each other
    :param state_1: One of the two NqueenStates about to undergo crossover with each other
    :return new_state_0, new_state_1: The pair of offspring produced by the crossover
    """

    board_0 = rotate_board(state_0.board.copy(), True)
    board_1 = rotate_board(state_1.board.copy(), True)

    random.seed()

    cut_point_0 = random.randint(0, min(int(len(board_0)/2), len(state_0.queens), len(state_1.queens)))
    cut_point_1 = random.randint(cut_point_0, min(len(board_0), len(state_0.queens), len(state_1.queens)))

    new_board_0 = board_1[:cut_point_0] + board_0[cut_point_0:cut_point_1] + board_1[cut_point_1:]
    new_board_1 = board_0[:cut_point_0] + board_1[cut_point_0:cut_point_1] + board_0[cut_point_1:]

    new_state_0 = NqueenState((state_0, state_1))
    new_state_1 = NqueenState((state_0, state_1))

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
    """
    Returns a new NqueenState with a blank board
    :param n: The specified board size
    :return state: The new NqueenState
    """

    random.seed()
    state = NqueenState(None)
    # state.set_board([[f'{i + (j * n)}' for i in range(n)] for j in range(n)])
    state.set_board([['-' for i in range(n)] for j in range(n)])
    # state.place_queen(random.randint(0, len(state.board) - 1))

    return state


def mutate_population(population):

    for mutation_index in range(len(population) - 1):
        random.seed()

        new_mutated_state = NqueenState()
        new_mutated_state.set_board(population[mutation_index].board)

        for i in range(len(new_mutated_state.board) - 1):

            complexTracker.increment_inner()

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
                nqueen_compute_weight(population[mutation_index])
                break
            else:
                None


def nqueen_heuristic_search(start, successorFunc, searchType, bigPop=0):

    biggest_pop = bigPop

    inner_iterations, outer_iterations, crossovers, mutations, goal = 0, 0, 0, 0, None

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
            # queens_left serves as simulated annealing scheduling variable
            # used alongside heuristic weight to determine the probability
            # that a successor that does not maximize cost is chosen when
            # the max isn't available.
            queens_left = (len(state.board) - len(state.queens))

            # Break loop and set return value if goal found
            if queens_left == 0 and state.cost >= 0:
                goal = state
                break

            # Expand the state and mark it as expanded if it hasn't been already
            if not expanded_states.search_expanded(state):
                state.successors = successorFunc(state)
                expanded_states.add_to_expanded(state)


            # If the state has successors
            if len(state.successors) > 0:

                # Determine the successor with the best weight
                costs = [successor.cost - queens_left for successor in state.successors]
                new_state = state.successors[costs.index(max(costs))]

                # If the
                if not expanded_states.search_expanded(new_state):
                    state = new_state
                elif len(state.successors) > 1:

                    temp_successors = state.successors.copy()
                    temp_successors.pop(costs.index(max(costs)))

                    temp_probs = [(float(successor.cost - state.cost) / (float(queens_left * 2)) + 0.1) * float(
                        not expanded_states.search_expanded(successor)) for successor in temp_successors]

                    try:
                        temp_probs = [prob/math.fsum(temp_probs) for prob in temp_probs]
                        random.seed()
                        successor_index = random.randint(1, 100)

                        for successor in temp_successors:
                            complexTracker.increment_inner()
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

            complexTracker.increment_outer()

    # Perform a genetic search
    if searchType == "genetic":

        population = [start]
        population_cap = int(len(start.board) / 2) + 1 if len(start.board) > 2 else len(start.board)

        # mutation_increment = (int(len(start.board) * 0.2) + 1)
        mutation_increment = 1
        mutation_timer = 0
        mutation_check = True
        best_rec_weight = -1 * len(start.board)
        wrst_best_rec_weight = -1 * len(start.board)

        # Loop until goal state found
        while goal is None:

            mutation_check = not mutation_check

            # Make a copy of the population so it can be modified
            temp_population = population.copy()

            for chromosome in temp_population:
                # inner_iterations += 1
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
                    # inner_iterations += 1
                    if f'{successor}' not in [f'{chromosome}' for chromosome in population]:
                        population.append(successor)

            # Break the loop if the goal was found
            if goal is not None:
                break

            # Sort population in ascending order of cost
            population.sort(key=lambda chromosome: chromosome.cost - (len(chromosome.board) - len(chromosome.queens)),
                            reverse=True)

            # Cull lower weight chromosomes to reduce pop size to its max allowed
            if len(population) > population_cap:

                population = population[:population_cap]

            # Calculate the best weight out of each chromosome in the population
            best_pop_weight = population[0].cost - (len(population[0].board) - len(population[0].queens))
            wrst_pop_weight = population[-1].cost - (len(population[-1].board) - len(population[-1].queens))

            # Set the base mutation probability to 10%
            mutation_probability = 0.1

            # If mutation timer increment reached and the best weight in the current population is greater
            # than the best recorded at the last increment
            if mutation_timer >= mutation_increment:
                # Reset the mutation timer
                mutation_timer = 0

                # The weights by which each aspect of the population impacts the probability that the population
                # will mutate
                best_prob_mod = 0.05
                biodiv_prob_mod = 0.05
                wrst_prob_mod = 0.005

                # Calculate the change in the best and worst fitness among the population compared to those of the
                # previously checked generation.
                delta_best = -1 * float(best_pop_weight - best_rec_weight)
                delta_wrst = -1 * float(wrst_pop_weight - wrst_best_rec_weight)

                # Calculate biodiversity by calculating the difference between the worst and best fitness among
                # the population.
                bio_div = float(best_pop_weight - wrst_pop_weight)

                # Calculate the amount by which the probability of the mutating will increase based on the change in
                # the best and worst fitness
                best_fitness_prob = (float(len(population)) * best_prob_mod) - (best_prob_mod * delta_best)
                wrst_fitness_prob = (float(len(population)) * wrst_prob_mod) - (wrst_prob_mod * delta_wrst)
                # wrst_fitness_prob = (float(len(population)) * wrst_prob_mod) - (wrst_prob_mod * delta_wrst)
                
                # Calculate the amount by which the mutation probability will increase based on the biodiversity among
                # the population.
                temp = float(len(population)) - bio_div
                biodiversity_prob = bio_div * biodiv_prob_mod

                if bio_div < 0.0:
                    x = 0

                # mutation_probability += (biodiversity_prob + best_fitness_prob + wrst_fitness_prob) * 100.0
                mutation_probability += (best_fitness_prob + wrst_fitness_prob) * 100.0

                if delta_best >= bio_div and bio_div > 0.0:
                    mutation_probability -= biodiversity_prob * 100

                if mutation_probability < 10:
                    mutation_probability = 10

                # If current best weight is better than the recorded best
                if best_pop_weight > best_rec_weight:
                    # Set best recorded weight to new best
                    best_rec_weight = best_pop_weight
                    # If current worst weight is better than the recorded worst
                    if wrst_pop_weight > wrst_best_rec_weight:
                        # Set threshold of worst weight to new population worst
                        wrst_best_rec_weight = wrst_pop_weight

            random.seed()
            genetic_operation = float(random.randint(1, 100))

            if genetic_operation < mutation_probability:
                complexTracker.increment_mutations()
                mutate_population(population)

            else:
                # Create a copy of the population so that population can be modified without affecting loops
                temp_population = population.copy()

                # Loop through entire population
                for chromosome in temp_population:

                    # Create copy of the temp population so it can be modified
                    other_chromosomes = temp_population.copy()

                    # Remove all occurrences of the current state (chromosome) from the population copy
                    for other_chromosome in temp_population:
                        complexTracker.increment_inner()
                        if check_boards_equal(chromosome, other_chromosome):
                            other_chromosomes.remove(other_chromosome)

                    # Loop until all chromosomes other than the current one are removed from the other list
                    while len(other_chromosomes) > 0:
                        complexTracker.increment_inner()
                        random.seed()

                        # Choose an index at random from the other population list
                        random_partner_index = random.randint(0, len(other_chromosomes) - 1)
                        # Pop and save the chosen chromosome
                        random_partner = other_chromosomes.pop(random_partner_index)
                        child_0, child_1 = genetic_crossover(chromosome, random_partner)
                        crossovers += 1

                        # Add each child to the list if they are not already in the population
                        if f'{child_0}' not in [f'{chromosome}' for chromosome in population]:
                            population.append(child_0)

                        if f'{child_1}' not in [f'{chromosome}' for chromosome in population]:
                            population.append(child_1)

                outer_iterations += 1

            # Sort population in ascending order of cost
            population.sort(key=lambda chromosome: chromosome.cost - (len(chromosome.board) - len(chromosome.queens)),
                            reverse=True)

            # Increment mutation timer
            mutation_timer += 1

    # Perform PSO on the problem
    if searchType == "PSO":
        # Set the best fitness to negative infinity
        best_fitness = float('-inf')
        # Initialize goal to None
        goal = None

        # The number of queens on each board
        num_queens = len(start.board)
        # Swarm size and max iterations determined by the size of the input (number of queens/columns)
        swarm_size = max(6, int(1.5 * num_queens))
        max_iters = num_queens * 125

        # Initialize a new random swarm
        swarm = initialize_swarm(num_queens, swarm_size)

        # Loop until max iterations reached
        for i in range(max_iters):
            # Save the particle with the best fitness in the
            best_particle = max(swarm, key=lambda particle: particle.cost)

            # Update the global best fitness and set the goal to be returned
            if best_particle.cost > best_fitness:
                best_fitness = abs(best_particle.cost)
                goal = best_particle

            # Break if the goal was found
            if best_fitness == 0:
                break

            # Update the macroscopic state
            mean_pos_board = update_macro_state(swarm)
            # Move the swarm to new positions
            swarm = advance_swarm(swarm, mean_pos_board, best_fitness)

            complexTracker.increment_outer()

    # Clear the list of expanded states
    expanded_states.clear_expanded()

    # Return the search results and reset the performance tracker
    result = (goal,
              (complexTracker.outer_iterations,
               complexTracker.inner_iterations,
               complexTracker.crossovers,
               complexTracker.mutations),
              biggest_pop
              )
    complexTracker.reset_all_counters()
    return result


def initialize_swarm(n, size):
    """
    Creates a new swarm of particles with randomized initial states. It generates a random permutation of integers
    representing the rows in which to place each queen, and then loops through each permutation placing each queen
    in its new space

    :param n: An int for the number of queens to be placed/number of columns or rows
    :param size: The number of particles in the swarm
    :return swarm: A newly initialized list of states where each queen is placed in each column in a random row
    """

    swarm_permutations = [np.random.permutation(n) for _ in range(size)]
    swarm = []

    for i, particle_permutation in enumerate(swarm_permutations):
        temp_state = nqueen_start(n)

        for queen_placement in particle_permutation:
            complexTracker.increment_inner()

            temp_state.successors = nqueen_successors(temp_state)
            temp_state = temp_state.successors[queen_placement]

        swarm.append(nqueen_start(n))
        swarm[i].set_board(temp_state.board)
        swarm[i].cost = temp_state.cost

    return swarm


def update_macro_state(swarm):
    """
    Updates and returns the macroscopic state of the swarm, which is represented by a 2D list of floats the same size
    as a state's board in which each float represents the fraction of states in the swarm that have a queen placed in
    that same cell.

    :param swarm: The list of states/particles that make up the swarm
    :return mean_board: A 2D list of floats
    """

    n = len(swarm[0].board)

    mean_board = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            complexTracker.increment_inner()

            queen_positions = [1 if (i, j) in particle.queens else 0 for particle in swarm]
            mean_value = stats.mean(queen_positions)
            mean_board[i][j] = mean_value

    return mean_board


def advance_swarm(swarm, mean_board, best_fit):
    """
    This function handles the movement of the swarm by modifying the boards of each particle in a manner determined
    probabilistically. The position of a particle is represented by the cost of its state and movement is
    represented by changes in the placements of its queens. The component of the velocity corresponding with the board
    space that is currently being observed is used as the probability that the queen in that column will be moved to
    that space.

    :param swarm: A list of nqueen states representing the swarm of particles
    :param mean_board: A 2D list of floats with the same dimensions as the board with each float being the fraction
        of particles with a queen placed in the corresponding space on the board
    :param best_fit: The best fitness out of any particle so far
    :return new_swarm: A list of nqueen states representing the newly repositioned swarm
    """

    new_swarm = []

    for index, particle in enumerate(swarm):
        n = len(particle.board)

        # Inertia and speeds/weights associated with each part of the velocity
        inertia, cognitive_wgt, social_wgt = 0.5, 0.9, 0.9

        new_particle = nqueen_start(n)
        new_particle.set_board(particle.board)
        new_particle.set_velocity()

        temp_velocity = new_particle.velocity.copy()

        for i in range(n):
            for j in range(n):
                # Update performance metrics
                complexTracker.increment_inner()

                # Calculate terms of velocity equation
                cognitive = cognitive_wgt * random.random() * (particle.personal_best - particle.cost)
                social = social_wgt * random.random() * mean_board[i][j] * (best_fit - particle.cost)

                # Update velocity
                temp_velocity[i][j] = inertia * new_particle.velocity[i][j] + cognitive + social

                # Determine whether to reposition the queen in the current column based on probability
                if random.random() < temp_velocity[i][j]:
                    new_particle.remove_queen(j)
                    new_particle.place_queen(i, j)

        new_swarm.append(nqueen_start(n))
        new_swarm[index].set_board(new_particle.board)
        new_swarm[index].set_velocity(temp_velocity)
        new_swarm[index].cost = nqueen_compute_weight(new_swarm[index])

    return new_swarm


# echo = False
echo = True
# echo_v = False
echo_v = True
# result_echo = False
result_echo = True

complexTracker.increment_outer()

biggest_pop = 0

overall_results = []
performance_results = []
performance_results_0 = []
solution_fitnesses = []
runtimes = []

tests = 150
n = 8

# test_types = ["hill-climbing", "genetic", "PSO"]
# test_types = ["genetic", "PSO"]
# test_types = ["hill-climbing", "genetic"]
# test_types = ["hill-climbing"]
# test_types = ["genetic"]
test_types = ["PSO"]

# Test Type Loop
for test_type in test_types:
    # Test Run Loop
    for i in range(tests):
        start_time = time.time()

        start_state = nqueen_start(n)
        search_result = nqueen_heuristic_search(start_state, nqueen_successors, test_type, biggest_pop)

        end_time = time.time()
        runtime = (end_time - start_time) * 1000
        runtimes.append(runtime)

        inn_iter_count = float(search_result[1][0])
        out_iter_count = float(search_result[1][1])
        mutations = search_result[1][3]
        crossovers = search_result[1][2]
        goal_state = search_result[0]
        biggest_pop = search_result[2]

        if echo_v and goal_state is not None:
            print(
                f'Test {i + 1}/{tests} (n = {n}) Goal:\n{goal_state}'
                f'cost: {goal_state.cost}\n{inn_iter_count} iterations\n'
                f'Runtime: {runtime:0.2f} ms\n')

            if test_type == "genetic":
                print(f'{mutations} mutations\n'
                      f'{crossovers} crossovers'
                      f'Runtime: {runtime:0.2f} ms\n')

        elif echo:
            print(f'Test {i + 1} finished\nRuntime: {runtime:0.2f} ms\n')

        if test_type == "genetic":
            performance_results.append(crossovers)
            performance_results_0.append(mutations)
        else:
            performance_results.append(inn_iter_count)
            performance_results_0.append(out_iter_count)

        solution_fitnesses.append(goal_state.cost)

    if result_echo:
        # Calculate the avg and standard deviation of innermost loop iterations that occurred during the searches
        avg_perf = stats.mean(performance_results)
        std_dev_perf = stats.stdev(performance_results)

        avg_perf_0 = stats.mean(performance_results_0)
        std_dev_perf_0 = stats.stdev(performance_results_0)

        # Calculate the avg and standard deviation of runtimes
        avg_runtime = stats.mean(runtimes)
        std_dev_runtime = stats.stdev(runtimes)

        # Calculate the average and standard deviation of fitness of the solutions produced by each test
        avg_solution_fitness = stats.mean(solution_fitnesses)
        std_dev_solution_fitness = stats.stdev(solution_fitnesses)

        overall_results.append((
            (avg_perf, std_dev_perf),
            (avg_perf_0, std_dev_perf_0),
            (avg_runtime, std_dev_runtime),
            (avg_solution_fitness, std_dev_solution_fitness)
        ))

    performance_results.clear()
    performance_results_0.clear()
    solution_fitnesses.clear()
    runtimes.clear()

print()
if result_echo:
    for i, test_type in enumerate(test_types):

        if str(test_type) == "genetic":
            print(f'Test: genetic (n = {n}):\n'
                  f' -\t# of tests: {tests}\n\n'
                  f' -\tAvg. Crossovers: {overall_results[i][0][0]}\n'
                  f' -\tStd. Dev. Crossovers: {overall_results[i][0][1]}\n\n'
                  f' -\tAvg. Mutations: {overall_results[i][1][0]}\n'
                  f' -\tStd. Dev. Mutations: {overall_results[i][1][1]}\n\n'
                  f' -\tAverage runtime: {overall_results[i][2][0]:.2f} ms\n'
                  f' -\tStd. Dev. Runtimes: {overall_results[i][2][1]:.2f} ms\n\n'
                  f' -\tAvg. Solution Fitness: {overall_results[i][3][0]:.2f} (Best possible fitness: 0)\n'
                  f' -\tStd. Dev. Fitness: {overall_results[i][3][1]:.2f}\n'
                  )
        else:
            print(f'Test: {test_type} (n = {n}):\n'
                  f' -\t# of tests: {tests}\n\n'
                  f' -\tAvg. Performance: {overall_results[i][0][0]} iterations\n'
                  f' -\tStd. Dev. Performance: {overall_results[i][0][1]}\n\n'
                  f' -\tAverage runtime: {overall_results[i][2][0]:.2f} ms\n'
                  f' -\tStd. Dev. Runtimes: {overall_results[i][2][1]:.2f} ms\n\n'
                  f' -\tAvg. Solution Fitness: {overall_results[i][3][0]:.2f} (Best possible fitness: 0)\n'
                  f' -\tStd. Dev. Fitness: {overall_results[i][3][1]:.2f}\n'
                  )

# probabilities = [0.5, 0.3, 0.1, 0.1]
# temp = sum(probabilities)
# selected_index = select_outcome(probabilities)
# print(f"Selected outcome index: {selected_index}")
#
# prob_results = []
# tests = 1000
#
# for i in range(tests):
#     temp = select_outcome(probabilities)
#
#     if temp != -1:
#         prob_results.append(temp)
#     else:
#         x = 0
#
# print(f'Frequencies:')
# for i in range(len(probabilities)):
#     print(f' - {i} ({probabilities[i]}): {float(prob_results.count(i)) / float(tests)}, Dev. From Prob: {probabilities[i] - (float(prob_results.count(i)) / float(tests))}')
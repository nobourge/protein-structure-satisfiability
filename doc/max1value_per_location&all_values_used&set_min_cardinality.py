import itertools

import numpy
from pysat.card import *
from pysat.card import ITotalizer
from pysat.solvers import Minisat22


def card(cnf
         , vpool
         , X
         , k
         ):
    print("card()")
    # , X, k)

    # for lit in X:
    # print("lit = ", lit)
    # print("vpool.id(lit) = ", vpool.id(lit))
    cnf.extend(CardEnc.atleast(lits=X
                               # cnf.append(CardEnc.atleast(lits=X
                               , bound=k
                               , vpool=vpool
                               , encoding=EncType.seqcounter
                               ))
    return cnf


def add_neighborhood_and_symbol_equivalence(to_append
                                            , vpool
                                            , x
                                            , y
                                            , sequence_index1
                                            , x2
                                            , y2
                                            , sequence_index2
                                            , neighborhood_symbol):
    # to_append.append([vpool.id((y
    #                             , x
    #                             , sequence_index1))
    #                      , -neighborhood_symbol
    #                      , -vpool.id((y2
    #                                   , x2
    #                                   , sequence_index2
    #                                   ))])
    # to_append.append([vpool.id((y2
    #                             , x2
    #                             , sequence_index2
    #                             ))
    #                      , -neighborhood_symbol
    #                      , -vpool.id((y
    #                                   , x
    #                                   , sequence_index1))
    #                   ])
    to_append.append([-neighborhood_symbol
                         , vpool.id((y
                                     , x
                                     , sequence_index1
                                     ))])
    to_append.append([-neighborhood_symbol
                         , vpool.id((y2
                                     , x2
                                     , sequence_index2
                                     ))])
    to_append.append([-vpool.id((y
                                 , x
                                 , sequence_index1
                                 ))
                         , -vpool.id((y2
                                      , x2
                                      , sequence_index2
                                      ))
                         , neighborhood_symbol])
    return to_append


def are_neighbors(i, j, k, m):
    # retourne True si et seulement si
    # (i, j) et (k, l) sont voisins
    # print("are_neighbors")
    if (abs(i - k) == 0 and
        abs(j - m) == 1) or \
            (abs(i - k) == 1 and
             abs(j - m) == 0):
        return True
    return False


def set_potential_neighbors_and_symbol(matrix_size,
                                       vpool,
                                       sequence_index1,
                                       sequence_index2
                                       , to_append
                                       , neighborhood_symbol=None
                                       # integer representing a locations pair neighborhood
                                       ,
                                       potential_neighbors_pairs_disjunctions_symbols=None
                                       ):
    print("set_potential_neighbors_and_symbol()")
    print("sequence_index1", sequence_index1)
    print("sequence_index2", sequence_index2)
    print("neighborhood_symbol", neighborhood_symbol)

    # deux points (i, j), (k, l) ∈ N² sont voisins si
    # (|i − k|, |j − l|) ∈ {(0, 1), (1, 0)}.
    pos_pairs = []
    for y in range(matrix_size):
        for x in range(matrix_size):
            for y2 in range(y - 1, y + 1):
                for x2 in range(x - 1, x + 1):
                    if y2 < 0 or y2 >= matrix_size \
                            or x2 < 0 or x2 >= matrix_size:
                        continue
                    # print("i = ", i
                    #       , "\nj = ", j
                    #       , "\nk = ", k
                    #       , "\nl = ", l)
                    if are_neighbors(y, x, y2, x2):
                        if (y, x, y2, x2) not in pos_pairs:
                            # pos_pairs.append((y, x, y2, x2))
                            pos_pairs.append((y2, x2, y, x))
                            # neighborhood_symbol <-> neighborhood

                            # neighborhood_symbol -> neighborhood &
                            # neighborhood_symbol <- neighborhood

                            # -neighborhood_symbol | neighborhood &
                            # neighborhood_symbol | neighborhood
                            add_neighborhood_and_symbol_equivalence(
                                to_append
                                , vpool
                                , y
                                , x
                                , sequence_index1
                                , y2
                                , x2
                                , sequence_index2
                                , neighborhood_symbol)

                            potential_neighbors_pairs_disjunctions_symbols.append(
                                neighborhood_symbol)
                            neighborhood_symbol += 1
                            #
                            # print("x, y, x2, y2", x, y, x2, y2)
                            #
                            # print("neighborhood_symbol",
                            #       neighborhood_symbol)
    return to_append, neighborhood_symbol


# return all sequence elements of specified value grouped by
# pair potential_neighborings_disjunction
def get_pairs_potential_neighborings_disjunctions_symbols(seq
                                                          ,
                                                          sequence_length
                                                          , cnf
                                                          , vpool
                                                          , matrix_size
                                                          , value=1
                                                          ):
    print("get_pairs_potential_neighborings_disjunctions_symbols()")
    # print("value = ", value)
    # potential_neighbors = [i for i in
    #                        range(sequence_length)]  # liste des voisins
    # # potentiels
    potential_neighbors = [0, 3]
    if 0 < len(potential_neighbors):
        print("potential_neighbors", potential_neighbors)
        # of desired value
        potential_neighbors_pairs_disjunctions_symbols = []
        neighborhood_symbol = 1
        # for index in potential_neighbors:
        #     for index2 in potential_neighbors:
        #         if index != index2 and \
        #                 (
        #                         index - index2) % 2 != 0:  # index and index2 are
        #             # separated by a pair
        #             # quantity of elements
        #
        #             # potential_neighbors_pairs_disjunctions_symbols \
        #             #     .append(neighborhood_symbol)
        #
        #             # create new set to append neighborhood without
        #             neighborhood_set = {}

        index = 0
        # index2 = 3
        index2 = 1

        cnf, neighborhood_symbol = \
            set_potential_neighbors_and_symbol(
                matrix_size
                ,
                vpool=vpool
                ,
                sequence_index1=index
                ,
                sequence_index2=index2
                , to_append=cnf
                ,
                neighborhood_symbol=neighborhood_symbol
                ,
                potential_neighbors_pairs_disjunctions_symbols=potential_neighbors_pairs_disjunctions_symbols
            )

        # index = 4
        # index2 = 7
        # cnf, neighborhood_symbol = \
        #     set_potential_neighbors_and_symbol(
        #         matrix_dimensions
        #         ,
        #         vpool=vpool
        #         ,
        #         sequence_index1=index
        #         ,
        #         sequence_index2=index2
        #         , to_append=cnf
        #         ,
        #         neighborhood_symbol=neighborhood_symbol
        #         ,
        #         potential_neighbors_pairs_disjunctions_symbols=potential_neighbors_pairs_disjunctions_symbols
        #     )
        # index = 1
        # index2 = 5
        # cnf, neighborhood_symbol = \
        #     set_potential_neighbors_and_symbol(
        #         matrix_dimensions
        #         ,
        #         vpool=vpool
        #         ,
        #         sequence_index1=index
        #         ,
        #         sequence_index2=index2
        #         , to_append=cnf
        #         ,
        #         neighborhood_symbol=neighborhood_symbol
        #         ,
        #         potential_neighbors_pairs_disjunctions_symbols=potential_neighbors_pairs_disjunctions_symbols
        #     )
        # index = 3
        # index2 = 6
        # cnf, neighborhood_symbol = \
        #     set_potential_neighbors_and_symbol(
        #         matrix_dimensions
        #         ,
        #         vpool=vpool
        #         ,
        #         sequence_index1=index
        #         ,
        #         sequence_index2=index2
        #         , to_append=cnf
        #         ,
        #         neighborhood_symbol=neighborhood_symbol
        #         ,
        #         potential_neighbors_pairs_disjunctions_symbols=potential_neighbors_pairs_disjunctions_symbols
        #     )
        return potential_neighbors_pairs_disjunctions_symbols
    return None


def set_min_cardinality(seq
                        , sequence_length
                        , cnf
                        , vpool
                        , value
                        , bound
                        , matrix_size
                        ):
    print("set_min_cardinality")
    pairs_potential_neighborings_disjunctions_symbols \
        = get_pairs_potential_neighborings_disjunctions_symbols(seq
                                                                ,
                                                                sequence_length
                                                                , cnf
                                                                , vpool
                                                                ,
                                                                matrix_size
                                                                , value
                                                                )
    if pairs_potential_neighborings_disjunctions_symbols is not None:
        if 1 < len(pairs_potential_neighborings_disjunctions_symbols):
            cnf = card(cnf
                       , vpool
                       ,
                       pairs_potential_neighborings_disjunctions_symbols
                       , bound
                       )
    else:
        print("No potential neighborings")
    return cnf


def all_values_used(sequence_length
                    , cnf
                    , vpool
                    , matrix_size):
    # print("All values in sequence must be in the answer")

    print("all_values_used()")

    txt = "clauses quantity:"
    print(f'{txt, cnf.nv}')
    print()

    for index in range(sequence_length):
        index_at_positions_disjunction = []
        for y in range(matrix_size):
            for x in range(matrix_size):
                index_at_positions_disjunction.append(vpool.id((y
                                                                , x
                                                                ,
                                                                index)))
        cnf.append(index_at_positions_disjunction)

    txt = "clauses quantity:"
    print(f'{txt, cnf.nv}')
    print()

    return cnf


def get_index_matrix(sequence_length
                     , matrix_size
                     , vpool
                     , sol):
    print("sequence_length", sequence_length)
    print("matrix_dimensions", matrix_size)
    matrix = numpy.matrix(numpy.zeros(shape=(matrix_size, matrix_size)))
    # matrix = [[0 for x in range(sequence_length)] for y in range(sequence_length)]
    # print("sequence_length", sequence_length)
    for index_i in range(matrix_size):
        for j in range(matrix_size):
            # print("i", index_i)
            # print("j", j)

            location_valued = False
            for v in range(sequence_length):
                # print("v: ", v)
                # print("type(v): ", type(v))
                if vpool.id((index_i, j, v)) in sol:
                    # print("v: ", v)
                    # print("type(v): ", type(v))
                    matrix[index_i, j] = v
                    location_valued = True
            if not location_valued:
                # print("Error: no value for location: ", index_i, j)
                matrix[index_i, j] = None
                # print(matrix)
            # print(matrix)

    return matrix


def get_value_matrix(matrix
                     , seq
                     , matrix_size):
    print("get_value_matrix")
    print("from matrix:")
    print(matrix)
    # value_matrix = [[0 for i in range(len(matrix))] for j in
    #                 range(len(matrix))]
    value_matrix = numpy.matrix(
        numpy.zeros(shape=(matrix_size, matrix_size)))

    for i in range(matrix_size):
        for j in range(matrix_size):
            # index = matrix[i][j]
            index = matrix[i, j]
            # print("index", index, "type : ", type(index))
            # print("seq index", seq[index], "type : ", type(seq[index]))

            # condition if index is numpy nan numpy.float64:
            if numpy.isnan(index):
                value_matrix[i, j] = None
                # value_matrix[i][j] = 3
            else:
                index = int(index)
                # print("index type : ", type(seq[index]))
                value_matrix[i, j] = seq[index]
    return value_matrix


def get_representation(value_matrix
                       ):
    matrix_size = len(value_matrix)
    representation = numpy.matrix(numpy.zeros(shape=(matrix_size,
                                                     matrix_size)
                                              , dtype=str))

    for i in range(matrix_size):
        for j in range(matrix_size):
            current = value_matrix[i, j]
            if numpy.isnan(current):
                representation[i, j] = " "
            else:
                representation[i, j] = str(current)
            # representation += str(value_matrix[i][j])
    # representation_colored = numpy.vectorize(get_color_coded_str)(
    #     value_matrix)
    # print("\n".join([" ".join(["{}"] * matrix_dimensions-1)] *
    #                 matrix_dimensions-1).format(*[x for y in
    #                                        representation_colored for x in y]))

    return representation


# si X_{x,y,i} alors non X_{x,y,i'}
# at most 1 value per cell
def max1value_per_location(sequence_length,
                           cnf,
                           vpool
                           , matrix_size
                           ):
    print("max1value_per_location")

    txt = "clauses quantity:"
    print(f'{txt, cnf.nv}')
    print()

    for i in range(matrix_size):
        for j in range(matrix_size):  # parcours tableau
            for index1 in range(sequence_length):
                for index2 in range(sequence_length):
                    if index1 != index2:
                        cnf.append([-vpool.id((i, j, index1)),
                                    -vpool.id((i, j, index2))])

    txt = "clauses quantity:"
    print(f'{txt, cnf.nv}')
    print()

    return cnf
#
# def max1location_per_value(sequence_length
#                            , cnf
#                            , vpool
#                            , matrix_dimensions):
#     print("max1location_per_value()")
#     for index in range(sequence_length):  # take 1 index
#         for x in range(matrix_dimensions):
#             for y in range(matrix_dimensions):  # take 1 cell
#                 for x2 in range(matrix_dimensions):
#                     for y2 in range(matrix_dimensions):  # take 2nd cell
#                         if not (x == x2 and
#                                 y == y2):
#                             # cell 1 and 2
#                             # can't have same index
#                             cnf.append([-vpool.id((x, y, index)),
#                                         -vpool.id((x2, y2, index))])
#     return cnf

def solve(seq,
          bound
          # , solver=Minisat22(use_timer=True)  # MiniSAT
          # # , solver=Glucose4(use_timer=True)  # MiniSAT
          ):
    # retourne un plongement de score au moins 'bound'
    # si aucune solution n'existe, retourne None

    sequence_length = len(seq)
    print("sequence_length", sequence_length)

    solver = Minisat22(use_timer=True)  # MiniSAT

    # variables ##########################
    vpool = IDPool(
        start_from=1)  # pour le stockage des identifiants entiers des couples (i,j)
    cnf = CNF()  # construction d'un objet formule en forme normale conjonctive (Conjunctive Normal Form)

    # contraintes ##########################
    matrix_size = sequence_length
    # matrix_dimensions = get_matrix_dimensions(sequence_length)
    print("matrix_dimensions", matrix_size)
    cnf = max1value_per_location(sequence_length, cnf, vpool,
                                 matrix_size)
    # cnf = max1location_per_value(sequence_length, cnf, vpool,
    #                              matrix_dimensions)
    cnf = all_values_used(sequence_length
                          , cnf
                          , vpool
                          , matrix_size)

    cnf = set_min_cardinality(seq
                              , sequence_length
                              , cnf
                              , vpool
                              , 1
                              , bound
                              , matrix_size)
    txt = "clauses quantity:"
    print(f'{txt, cnf.nv}')
    print()
    # print("cnf", cnf)
    # print("cnf clauses", cnf.clauses)

    # solver = Glucose4(use_timer=True)
    solver.append_formula(cnf.clauses, no_return=False)

    print("Resolution...")
    resultat = solver.solve()
    # print("seq ", seq)
    # print("bound ", bound)
    print("Satisfaisable : " + str(resultat))
    print("Temps de resolution : " + '{0:.2f}s'.format(solver.time()))
    if resultat:
        interpretation = get_interpretation(solver
                                            )
        index_matrix = get_index_matrix(sequence_length
                                        , matrix_size
                                        , vpool
                                        , interpretation
                                        )
        value_matrix = get_value_matrix(index_matrix
                                        , seq
                                        , matrix_size)
        display = True
        if display:
            print("\nVoici une solution: \n")
            print(get_representation(value_matrix
                                     ))
            return value_matrix
    return None


def get_interpretation(solver
                       ):
    interpretation = solver.get_model()  # extracting a
    # satisfying assignment for CNF formula given to the solver
    # A model is provided if a previous SAT call returned True.
    # Otherwise, None is reported.
    # Return type list(int) or None

    print("interpretation", interpretation)

    return interpretation

#
solve("00", 0)
# get_solution_representation("100010100", 0)
solve("111", 1)
solve("1001", 1)
solve("111001", 1)
solve("11100000001", 4)
solve("1110000000111111", 4)

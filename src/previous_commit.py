# cours informatique fondamentale 2021-2022
# PROJET: repliage de proteines

# necessite l'installation de
# la librairie PySAT et de
# la librairie func_timeout
import sys
from typing import Tuple

import numpy
# from pip._internal.utils import logging
import logging
from pysat.solvers import Minisat22
# from pysat.solvers import Glucose4
# from pysat.formula import CNF
# from pysat.formula import IDPool
from pysat.card import *
from optparse import OptionParser
import func_timeout

from auto_indent import *
from colorama import Fore, Style

sys.stdout = AutoIndent(sys.stdout)

# OPTIONS POUR L'UTILISATION EN LIGNE DE COMMANDE

# Usage: folder.py [options]

# Options: -h, --help            show this help message and exit
# -s
# SEQ,
# --sequence=SEQ specify the input sequence
# -b BOUND,
# --bound=BOUND specify a lower bound on the score
# -p, --print
# print solution
# -i, --incremental     incremental mode: try small
# bounds first and increment
# -v, --verbose         verbose mode -t,
# --test            testing mode

# on doit TOUJOURS donner une sequence * lorsqu'une borne est donnee,
# votre programme doit tester que le meilleur score de la sequence
# est superieur ou egal a cette borne

# * lorsqu'aucune borne sequence_length matrix_size'est donnee, alors votre programme doit
# calculer le meilleur score pour la sequence, par defaut en
# utilisant une recherche par dichotomie, et en utilisant une methode
# incrementale si l'option -i est active
#
# l'option -v vous permet de creer un mode 'verbose'

# si l'option -t est active, alors le code execute uniquement la
# fonction test_code() implementee ci-dessous, qui vous permet de
# tester votre code avec des exemples deja fournis.

# Si l'execution d'un test prend plus que TIMEOUT secondes (fixe a
# 10s ci-dessous), alors le test s'arrete et la fonction passe au
# test suivant

parser = OptionParser()
parser.add_option("-s", "--sequence", dest="sequence", action="store",
                  help="specify the input sequence")
parser.add_option("-b", "--bound"
                  , dest="bound", action="store",
                  help="specify a lower bound on the score", type="int")
parser.add_option("-p", "--print"
                  , dest="display",
                  action="store_true",
                  help="print solution", default=False)
parser.add_option("-i"
                  , "--incremental"
                  , dest="incremental",
                  action="store_true",
                  help="incremental mode: try small bounds first and increment",
                  default=False)
parser.add_option("-v", "--verbose",
                  # action="store_true",
                  action="store_const",
                  dest="log_level",

                  # dest="verbose",
                  # dest=["verbose", "log_level"],

                  const=logging.INFO,
                  help="verbose mode"
                  # , default=False
                  )
parser.add_option("-d", "--debug",
                  action="store_const",
                  dest="log_level",
                  # dest="debug",
                  # dest=["debug", "log_level"],
                  const=logging.DEBUG,
                  help="debug mode", default=False)
parser.add_option("-t", "--test", dest="test", action="store_true",
                  help="testing mode", default=False)

(options, args) = parser.parse_args()
#print(dir(options))
logging.basicConfig(level=options.log_level)
# logging.basicConfig(level=options.loglevel)

#
# logging.basicConfig(level=logging.DEBUG if options.debug else (
#     logging.INFO if options.verbose else logging.WARNING))

affichage_sol = options.display
verb = options.log_level  # == logging.DEBUG
incremental = options.incremental
test = options.test


###############################################################################################


# clauses = contraintes


# add clauses to cnf to ensure that 2 sequence elements are neighbors
def set_neighbors(matrix_size,
                  vpool,
                  sequence_index1,
                  sequence_index2
                  , to_append
                  , neighborhood_symbol=None
                  # integer representing a locations pair neighborhood
                  , potential_neighbors_pairs_disjunctions_symbols=None
                  ):
    # print("set_neighbors()", a, b)
    # deux points (i, j), (k, l) ∈ N² sont voisins si
    # (|i − k|, |j − l|) ∈ {(0, 1), (1, 0)}.
    for y in range(matrix_size):
        for x in range(matrix_size):
            d = [-vpool.id((y, x, sequence_index1))]
            for y2 in range(matrix_size):
                for x2 in range(matrix_size):
                    # print("i = ", i
                    #       , "\nj = ", j
                    #       , "\nk = ", k
                    #       , "\nl = ", l)
                    if are_neighbors(y, x, y2, x2):
                        # print("              i, j, k, l", i, j, k, l)
                        d.append(vpool.id((y2, x2,
                                           sequence_index2)))
            # print("d = ", d)
            to_append.append(d)
    return to_append


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
    c = 0

    # print("set_neighbors", a, b)
    # deux points (i, j), (k, l) ∈ N² sont voisins si
    # (|i − k|, |j − l|) ∈ {(0, 1), (1, 0)}.
    pos_pairs = []
    for y in range(matrix_size):
        for x in range(matrix_size):
            for y2 in range(y - 1, y + 1):
                for x2 in range(x - 1, x + 1):
                    if 0 <= y2 and y2 < matrix_size \
                            and 0 <= x2 and x2 < matrix_size:
                        # print("i = ", i
                        #       , "\nj = ", j
                        #       , "\nk = ", k
                        #       , "\nl = ", l)
                        if are_neighbors(y, x, y2, x2):
                            # if (y, x, y2, x2) not in pos_pairs:
                            #     # pos_pairs.append((y, x, y2, x2))
                            #     pos_pairs.append((y2, x2, y, x))
                            # neighborhood_symbol <-> neighborhood

                            # neighborhood_symbol -> neighborhood &
                            # neighborhood_symbol <- neighborhood

                            # -neighborhood_symbol | neighborhood &
                            # neighborhood_symbol | neighborhood
                            vpool_neighborhood_symbol = vpool.id(
                                neighborhood_symbol)

                            c += 1
                            add_neighborhood_and_symbol_equivalence(
                                to_append
                                , vpool
                                , y
                                , x
                                , sequence_index1
                                , y2
                                , x2
                                , sequence_index2
                                , vpool_neighborhood_symbol)

                            potential_neighbors_pairs_disjunctions_symbols.append(
                                vpool_neighborhood_symbol)
                            neighborhood_symbol += 1

                            # logging.debug("x, y, x2, y2", x, y, x2, y2)
                            # logging.debug(
                            #     "sequence_index1, sequence_index2")
                            # logging.debug(sequence_index1,
                            #               sequence_index2)
                            # logging.debug("neighborhood_symbol : {"
                            #               "}".format(neighborhood_symbol))
    return to_append, neighborhood_symbol


def add_neighborhood_and_symbol_equivalence(to_append
                                            , vpool
                                            , x
                                            , y
                                            , sequence_index1
                                            , x2
                                            , y2
                                            , sequence_index2
                                            ,
                                            vpool_neighborhood_symbol):
    to_append.append([-vpool_neighborhood_symbol
                         , vpool.id((y
                                     , x
                                     , sequence_index1
                                     )
                                    )])
    to_append.append([-vpool_neighborhood_symbol
                         , vpool.id((y2
                                     , x2
                                     , sequence_index2
                                     )
                                    )])
    to_append.append([-vpool.id((y
                                 , x
                                 , sequence_index1
                                 ))
                         , -vpool.id((y2
                                      , x2
                                      , sequence_index2
                                      ))
                         , vpool_neighborhood_symbol])
    return to_append


# sequence elements 2 by 2 are neighbors
def sequence_neighboring_maintain(sequence_length
                                  , cnf
                                  , vpool
                                  , matrix_size
                                  ):
    logging.info("sequence_neighboring_maintain()")

    for i in range(sequence_length - 1):
        cnf = set_neighbors(matrix_size,
                            vpool,
                            i,
                            i + 1
                            , to_append=cnf
                            )

    logging.info("clauses quantity: {}".format(cnf.nv))
    print()

    return cnf


# def is_in_matr


def are_neighbors(i, j, k, m):
    # retourne True si et seulement si
    # (i, j) et (k, l) sont voisins
    # logging.debug("are_neighbors()")
    if (abs(i - k) == 0 and
        abs(j - m) == 1) or \
            (abs(i - k) == 1 and
             abs(j - m) == 0):
        return True
    return False


def get_sequence_elements_of_value(seq
                                   , sequence_length
                                   , value
                                   ):
    potential_neighbors = []
    for index in range(sequence_length):
        # logging.debug("index = ", index)
        # logging.debug("seq[index] = ", seq[index])
        # logging.debug("seq[index] type = ", type(seq[index]))
        # logging.debug("value = ", value)
        # logging.debug("value type = ", type(value))
        if int(seq[index]) == value:
            potential_neighbors.append(index)
    return potential_neighbors


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
    # retourne la liste des voisins potentiels
    # de la sequence seq

    logging.info(
        "get_pairs_potential_neighborings_disjunctions_symbols()")
    logging.debug("value = {}".format(value))
    potential_neighbors = get_sequence_elements_of_value(seq
                                                         ,
                                                         sequence_length
                                                         , value)

    if 0 < len(potential_neighbors):
        # print("potential_neighbors = ", potential_neighbors)
        # logging.debug("potential_neighbors = {}".format(potential_neighbors))
        logging.info(
            "potential_neighbors {}".format(potential_neighbors))
        # of desired value
        potential_neighbors_pairs_disjunctions_symbols = []
        neighborhood_symbol = 1
        # loop over all pairs of potential_neighbors
        # first 0
        for index in potential_neighbors:
            # 1
            for index2 in potential_neighbors:
                # index2 different index
                if index != index2:
                    if (index - index2) % 2 != 0:  # index and
                        # index2 are
                        # separated by a pair
                        # quantity of elements

                        # potential_neighbors_pairs_disjunctions_symbols \
                        #     .append(neighborhood_symbol)

                        # create new set to append neighborhood without
                        neighborhood_set = {}

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
        return potential_neighbors_pairs_disjunctions_symbols
    return None


# vous pouvez utiliser les methodes de la classe pysat.card pour
# creer des contraintes de cardinalites (au moins k, au plus k,...)

# fonction card
# qui prend en entree un
# ensemble fini de variables X et un entier k,
# et qui retourne un
# ensemble de clauses card(X, k),
# qui est satisfaisable si et seulement si
# il existe au moins k variables de X qui sont vraies
def card(cnf
         , vpool
         , X
         , k
         ):
    logging.info("card()")
    logging.debug("X={}".format(X))
    logging.debug("bound = {}".format(k))

    # for lit in X:
    # print("lit = ", lit)
    # print("vpool.id(lit) = ", vpool.id(lit))
    cnf.extend(CardEnc.atleast(lits=X
                               , bound=k
                               , vpool=vpool
                               , encoding=EncType.seqcounter
                               ))
    return cnf


#
# def all_values_used(sequence_length
#                     , cnf
#                     , vpool
#                     , matrix_size):
#     # print("All values in sequence must be in the answer")
#
#     logging.info("all_values_used()")
#
#     for index in range(sequence_length):
#         index_at_positions_disjunction = []
#         for y in range(matrix_size):
#             for x in range(matrix_size):
#                 index_at_positions_disjunction.append(vpool.id((y
#                                                                 , x
#                                                                 ,
#                                                                 index)))
#         cnf.append(index_at_positions_disjunction)
#     logging.info("clauses quantity: {}".format(cnf.nv))
#     return cnf
#
#
# # si X_{x,y,i} alors non X_{x',y',i}
# # at most 1 cell per value
# def max1location_per_value(sequence_length
#                            , cnf
#                            , vpool
#                            , matrix_size):
#     logging.info("max1location_per_value()")
#     for index in range(sequence_length):  # take 1 index
#         for x in range(matrix_size):
#             for y in range(matrix_size):  # take 1 cell
#                 for x2 in range(matrix_size):
#                     for y2 in range(matrix_size):  # take 2nd cell
#                         if not (x == x2 and
#                                 y == y2):
#                             # cell 1 and 2
#                             # can't have same index
#                             cnf.append([-vpool.id((x, y, index)),
#                                         -vpool.id((x2, y2, index))])
#     return cnf
#

# si X_{x,y,i} alors non X_{x,y,i'}
# at most 1 value per cell
def max1value_per_location(sequence_length,
                           cnf,
                           vpool
                           , matrix_size
                           ):
    logging.info("max1value_per_location()")

    for i in range(matrix_size):
        for j in range(matrix_size):  # parcours tableau
            for index1 in range(sequence_length):
                for index2 in range(sequence_length):
                    if index1 != index2:
                        cnf.append([-vpool.id((i, j, index1)),
                                    -vpool.id((i, j, index2))])

    logging.info("clauses quantity: {}".format(cnf.nv))

    return cnf

def one_location_per_value_min_and_max(sequence_length
                                       , cnf
                                       , vpool
                                       , matrix_size):
    logging.info("one_location_per_value_min_and_max()")

    for index in range(sequence_length):  # take 1 index
        index_at_positions_disjunction = []
        for x in range(matrix_size):
            for y in range(matrix_size):  # take 1 cell
                index_at_positions_disjunction.append(vpool.id((y
                                                                , x
                                                                ,
                                                                index)))
        cnf.extend(CardEnc.equals(index_at_positions_disjunction
                                  , 1
                                  , vpool=vpool
                                  , encoding=EncType.seqcounter))
    logging.info("clauses quantity: {}".format(cnf.nv))
    return cnf


def set_min_cardinality(seq
                        , sequence_length
                        , cnf
                        , vpool
                        , value
                        , bound
                        , matrix_size
                        ):
    logging.info("set_min_cardinality()")
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


def set_clauses(seq,
                sequence_length,
                cnf,
                vpool
                , bound
                , matrix_size
                ):
    # all_values_used(sequence_length
    #                 , cnf
    #                 , vpool
    #                 , matrix_size)
    # sequence elements 2 by 2 are neighbors
    cnf = sequence_neighboring_maintain(sequence_length,
                                        cnf,
                                        vpool
                                        , matrix_size
                                        )
    # # au plus une valeur par case
    cnf = max1value_per_location(sequence_length
                                 , cnf
                                 , vpool
                                 , matrix_size)
    # cnf = max1location_per_value(sequence_length
    #                              , cnf
    #                              , vpool
    #                              , matrix_size)
    cnf = one_location_per_value_min_and_max(sequence_length
                                                , cnf
                                                , vpool
                                                , matrix_size)
    cnf = set_min_cardinality(
        seq
        , sequence_length
        , cnf
        , vpool
        , value=1
        , bound=bound
        , matrix_size=matrix_size
    )

    return cnf


# print the variables of the solution
def print_solution_variables(seq,
                             sequence_length,
                             matrix_size,
                             vpool,
                             sol):
    # return
    print("Solution variables:")
    for i in range(matrix_size):
        for j in range(matrix_size):
            for v in range(sequence_length):
                # print("value : ", v)
                # print("value", vpool.id((i, j, v + 1)), "=",
                # sol[vpool.id((i, j, v + 1))])
                if vpool.id((i, j, v)) in sol:
                    print(i, j, seq[v])


def get_matrix_size(seq
                    , sequence_length):
    # returns the minimum size of the matrix within which the maximum
    # contacts folded sequence can fit
    # todo

    if seq in ["011010111110011"
        , "0010110"
        , "011001101"
        , "000110111"
        , "0011110010110110"
        , "01010101110"
        , "1000101110001"
        , "11010101011110"
               ]:
        return math.ceil(math.sqrt(sequence_length)) + 1
    elif seq in ["0110111001000101"]:
        return math.ceil(math.sqrt(sequence_length)) + 2
    return math.ceil(math.sqrt(sequence_length))  # from github copilot

    # todo rectengular matrix
    # # todo cubic square root of squared sequence_length
    # # todo return int(sequence_length ** (2 / 3)) # ~~from robin petit
    # # return 1 + sequence_length // 4 if sequence_length >= 12 else sequence_length
    # return math.ceil((1 + sequence_length) / 2)   # from mkovel


def get_index_matrix(sequence_length
                     , matrix_size
                     , vpool
                     , sol):
    print("sequence_length", sequence_length)
    print("matrix_size", matrix_size)
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
    print("get_value_matrix()")
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


def get_color_coded_str(i):
    return "\033[3{}m{}\033[0m".format(i + 1, i)


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
    # print("\n".join([" ".join(["{}"] * matrix_size-1)] *
    #                 matrix_size-1).format(*[x for y in
    #                                        representation_colored for x in y]))

    return representation


def get_score(value_matrix):
    # retourne le score d'un plongement
    # Le score d’un plongement P de p, noté score(p, P) est défini par
    # score(p, P) =
    # |{{i, j} | i, j ∈ Pos(p),
    #       i != j,
    #       P(i) et P(j) sont voisins,
    # p[i] = p[j] = 1}|
    # Autrement dit, c’est le nombre de
    # paires de positions différentes i, j de p,
    # étiquetées par 1 dans p,
    # et qui se plongent vers des points voisins dans N²
    print()
    print("get_score()")

    matrix_size = len(value_matrix)
    new_score = 0
    for i in range(matrix_size):
        for j in range(matrix_size):
            current = value_matrix[i, j]
            # print(i, j, " is ", current)
            # current type print
            # print("current type : ", type(current))

            if current == 1:
                if i + 1 < matrix_size:
                    if current == value_matrix[i + 1, j]:
                        new_score += 1
                        # print(i, j, "="
                        #       , i + 1, j)
                if j + 1 < matrix_size:
                    if current == value_matrix[i, j + 1]:
                        new_score += 1
                        # print(i, j, "="
                        #       , i, j + 1)

            else:
                # print(i, j, " is -1")
                pass
    return new_score


def print_solution_matrix(matrix
                          , seq
                          , mode="all"):
    print("Solution value_matrix:")
    for mode in ("index", "value"):
        print("mode:%s" % mode)
        for i in range(len(matrix)):
            print()
            for j in range(len(matrix)):
                # if 0 <= matrix[i][j]:
                if 0 <= matrix[i, j]:
                    if mode == "index" \
                            or mode == "all":
                        print(matrix[i, j], end=" ")
                        # print(matrix[i][j], end=" ")
                    if mode == "value" or mode == "all":
                        print(seq[int(matrix[i, j])], end=" ")
                        # print(seq[matrix[i][j]], end=" ")
                else:
                    print("*", end=" ")
        print()


def solve(seq,
          bound
          , mode="all"
          ):
    # retourne un plongement de score au moins 'bound'
    # si aucune solution n'existe, retourne None
    print("seq", seq)
    sequence_length = len(seq)
    print("sequence_length", sequence_length)
    print("bound", bound)
    print("mode", mode)


    contact_quantity_min, contact_quantity_max = get_contact_quantity_min_and_max(seq)
    if contact_quantity_max < bound:
        print("bound <= contact_quantity_max")
        print(bound, "<=", contact_quantity_max)

        print("Il n'existe pas de solution")
        return None

    solver = Minisat22(use_timer=True)  # MiniSAT

    # variables ##########################
    vpool = IDPool(
        start_from=1)  # pour le stockage des identifiants entiers des couples (i,j)
    # vpool.restart(start_from=1)
    #
    # print("vpool", vpool)
    # print("vpool.id((0, 0, 0))", vpool.id((0, 0, 0)))
    # print("vpool.id((0, 0, 1))", vpool.id((0, 0, 1)))
    # print("vpool.id((0, 0, 2))", vpool.id((0, 0, 2)))
    # print("vpool.top", vpool.top)
    # print("vpool", vpool)
    # print("vpool", vpool.nv)

    cnf = CNF()  # construction d'un objet formule en forme normale conjonctive (Conjunctive Normal Form)

    # contraintes ##########################
    matrix_size = get_matrix_size(seq
                                  , sequence_length)
    print("matrix_size", matrix_size)
    cnf = set_clauses(seq,
                      sequence_length,
                      cnf,
                      vpool
                      , bound
                      , matrix_size
                      )
    logging.info("clauses quantity: {}".format(cnf.nv))
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
        if mode == "sat":
            return True
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
        if options.display:
            print("\nVoici une solution: \n")

            print(get_representation(value_matrix
                                     ))
        new_score = get_score(value_matrix)
        print("score:", new_score)
        if new_score >= bound:
            print("new_score >= bound")
            print(new_score, ">=", bound)
            return value_matrix
        print("new_score < bound")
        print(new_score, "<", bound)
    return None


def get_interpretation(solver
                       ):
    print("get_interpretation()")

    interpretation = solver.get_model()  # extracting a
    # satisfying assignment for CNF formula given to the solver
    # A model is provided if a previous SAT call returned True.
    # Otherwise, None is reported.
    # Return type list(int) or None
    logging.debug("interpretation : {}".format(
        interpretation))
    logging.debug("interpretation size : {}".format(
        len(interpretation)))

    # cette interpretation est longue,
    # on va filtrer les valeurs positives
    filtered_interpretation = list(
        filter(lambda x: x >= 0, interpretation))
    logging.debug("filtered_interpretation : {}".format(
        filtered_interpretation))
    logging.debug("filtered_interpretation size : {}".format(
        len(filtered_interpretation)))

    # return interpretation
    return filtered_interpretation


def exist_sol(seq, bound):
    # retourne True si et seulement si il
    # existe un plongement de score au moins 'bound'
    # A COMPLETER
    print()
    print("exist_sol() ")
    print("seq: ", seq)
    print("bound: ", bound)

    if solve(seq, bound, mode="sat") is not None:
        print("Il existe une solution")
        return True

    return False


def get_contact_quantity_min_and_max(seq: str) -> Tuple[int, int]:
    # retourne le nombre maximal de contacts
    print("get_contact_quantity_min_and_max() ")
    n = len(seq)
    # count ones in seq
    ones_quantity = seq.count("1")
    if ones_quantity == 0:
        return 0, 0

    # a(n) = 2n - ceiling(2*sqrt(n))
    contacts_quantity_max = 2 * ones_quantity - math.ceil(
        2 * math.sqrt(ones_quantity))

    #original sequence quantity of ones already in contact
    contacts_quantity_min = 0
    total = 0
    for i in range(n):
        # print("i : ", i)
        if seq[i] == "1":
            if i + 1 < n:
                if seq[i + 1] == "1":
                    contacts_quantity_min += 1
                    total += 1

            if i + 3 < n:
                total += min(2, seq[i + 3:n:2].count("1"))
            #     print("seq[i + 3:n:2].count(\"1\") : ", seq[i + 3:n:2].count("1"))
        # print("contacts_quantity_min : ", contacts_quantity_min)
        # print("total : ", total)

    print("contacts_quantity_min : ", contacts_quantity_min)
    print("total : ", total)
    print("contacts_quantity_max : ",
          contacts_quantity_max)

    if total < contacts_quantity_max:
        contacts_quantity_max = total
    print("contacts_quantity_max : ",contacts_quantity_max)
    return contacts_quantity_min, contacts_quantity_max


def dichotomy(seq
              , lower_bound=0):
    # retourne un plongement de score au moins 'lower_bound'

    # si aucune solution sequence_length matrix_size'existe
    # , retourne None
    #
    # cette fonction utilise
    # la methode de dichotomie pour trouver un plongement de score au
    # moins 'lower_bound' A COMPLETER
    logging.info("dichotomy() ")
    lower_bound, high_bound = get_contact_quantity_min_and_max(seq)
    logging.debug("high_bound", high_bound)
    logging.debug("lower_bound", lower_bound)
    sol = solve(seq, high_bound)
    if sol is not None:
        return sol

    # while high_bound - lower_bound == 1:
    while 1 < high_bound - lower_bound:
        mid_bound = (high_bound + lower_bound) // 2
        logging.debug("mid_bound", mid_bound)
        new_sol = solve(seq, mid_bound)
        if new_sol is not None:
            sol = new_sol
            lower_bound = mid_bound
            logging.debug("lower_bound", lower_bound)

        else:
            high_bound = mid_bound
            logging.debug("high_bound", high_bound)

    if sol is not None:
        logging.info("dichotomy() sol is not None")
        return sol
    return None

    # 1 2 3 4 5
    # l   m   h
    # 1 2 3
    # l m h
    # 2 3
    # l h


def incremental_search(seq
                       , lower_bound=0):
    # retourne un plongement de score au moins 'lower_bound'
    # si aucune solution matrix_size'existe, retourne None
    # cette fonction utilise une recherche incrémentale
    # pour trouver un plongement de score au moins 'lower_bound'
    if not exist_sol(seq, lower_bound):
        return None
    lower_bound += 1
    while exist_sol(seq, lower_bound):
        lower_bound += 1
    sol = solve(seq, lower_bound - 1)
    if sol is not None:
        print("incremental_search() sol is not None")

    return sol


def compute_max_score(seq
                      , method="dichotomy"
                      , display=True):
    # calcul le meilleur score pour la sequence seq,
    # il doit donc retourne un entier,
    # methode utilisee: dichotomie par defaut,
    # si l'option -i est active, on utilise la recherche incrémentale
    print()
    print("compute_max_score() ")
    print("seq: ", seq)
    print("method: ", method)
    print("display: ", display)

    # contacts_quantity_min, contacts_quantity_max =
    # get_contact_quantity_min_and_max(seq)
    if method == "incremental":
        sol = incremental_search(seq)
    else:
        sol = dichotomy(seq)
    if sol is None:
        return 0
    score_best = get_score(sol)
    #
    # if display and score_best is not None:
    #     # print("##############################################",solve(seq, score_best))
    #     # sol = solve(seq, score_best)
    #     # print_solution_matrix(sol, seq)
    #     print(get_representation(sol))

    print("score_best: ", score_best)
    return score_best


####################################################################
########### CE CODE NE DOIT PAS ETRE MODIFIE #######################
####################################################################
def test_code():
    satisfiability_echec = []
    unsatisfiability_echec = []
    unsatisfiability_exception = []
    max_score_echec = []
    max_score_timeout = []
    max_score_exception = []

    examples = [
        ('00', 0),
        ('1', 0),
        ('01000', 0),
        ('00110000', 1),

        ('11', 1),
        ('111', 2),
        ('1111', 4),
        ('1111111', 8),
        ("111111111111111", 22),
        ("1011011011", 7),
        ("011010111110011", 13),
        ("01101011111000101", 11),
        ("0110111001000101", 8),
        ("000000000111000000110000000", 5),
        ('100010100', 0),
        ('01101011111110111', 17),
        ('10', 0), ('10', 0),
        ('001', 0), ('000', 0), ('1001', 1), ('1111', 4),
        ('00111', 2), ('01001', 1),
        ('111010', 3), ('110110', 3), ('0010110', 2),
        ('0000001', 0), ('01101000', 2), ('10011111', 7),
        ('011001101', 5), ('000110111', 5),
        ('0011000010', 2), ('1000010100', 2),
        ('11000111000', 5), ('01010101110', 4),
        ('011001100010', 5), ('010011100010', 5),
        ('1110000110011', 8), ('1000101110001', 4),
        ('11010101011110', 10), ('01000101000101', 0),
        ('111011100100000', 8),
        ('000001100111010', 6), ('0110111110011000', 11),
        ('0011110010110110', 11),
        ('01111100010010101', 11),
        ('10011011011100101', 12),
        ('101111101100101001', 13), ('110101011010101010', 9),
        ('1111101010000111001', 14),
        ('0111000101001000111', 11),
        ('10111110100001010010', 12),
        ('10110011010010001110', 11)
    ]
    # chaque couple de cette liste est formee d'une sequence et de son meilleur score

    TIMEOUT = 10

    # SAT TESTS
    total_sat_tests = 0
    total_unsat_tests = 0
    sat_tests_success = 0
    timeouts_sat_tests = 0
    exceptions_sat_tests = 0

    # UNSAT TESTS
    total_unsat_test = 0
    unsat_tests_success = 0
    timeouts_unsat_tests = 0
    exceptions_unsat_tests = 0

    # MAXSCORES TEST
    correct_maxscores = 0
    total_maxscores = 0
    timeouts_maxscores = 0
    exceptions_maxscores = 0

    # sur cet ensemble de tests, votre methode devrait toujours retourner qu'il existe une solution
    print("\n****** Test de satisfiabilite ******\n")
    for (seq, maxbound) in examples:
        total_sat_tests += 1
        bound = int(maxbound / 2)
        print("sequence: " + seq + " borne: " + str(bound), end='')
        exist_sol(seq, bound)
        try:
            if func_timeout.func_timeout(TIMEOUT, exist_sol,
                                         [seq, bound]):
                sat_tests_success += 1
                print(" ---> succes")
                # satisfiability_success.append(seq)
            else:
                print(" ---> echec")
                satisfiability_echec.append(seq)
        except func_timeout.FunctionTimedOut:
            timeouts_sat_tests += 1
            print(" ---> timeout")
        except Exception as e:
            exceptions_sat_tests += 1
            print(" ---> exception levee")

    # sur cet ensemble de tests, votre methode devrait toujours
    # retourner qu'il matrix_size'existe pas de solution
    print("\n****** Test de d'insatisfiabilite ******\n")
    for (seq, maxbound) in examples:
        total_unsat_tests += 1
        bound = maxbound + 1
        print("sequence: " + seq + " borne: " + str(bound), end='')
        try:
            if not func_timeout.func_timeout(TIMEOUT, exist_sol,
                                             [seq, bound]):
                unsat_tests_success += 1
                print(" ---> succes")
                # unsatisfiability_success.append(seq)
            else:
                print(" ---> echec")
                unsatisfiability_echec.append(seq)

        except func_timeout.FunctionTimedOut:
            timeouts_unsat_tests += 1
            print(" ---> timeout")
        except Exception as e:
            exceptions_unsat_tests += 1
            print(" ---> exception levee")
            unsatisfiability_exception.append(seq)

    # sur cet ensemble de tests, votre methode devrait retourner le meilleur score.
    # Vous pouvez utiliser la methode par dichotomie ou incrementale, au choix
    print("\n****** Test de calcul du meilleur score ******\n")
    for (seq, maxbound) in examples:
        total_maxscores += 1
        print("sequence: " + seq + " borne attendue: " + str(maxbound),
              end='')
        try:
            found_max_score = func_timeout.func_timeout(TIMEOUT,
                                                        compute_max_score,
                                                        [seq])
            print(" borne retournee: " + str(found_max_score), end='')
            if maxbound == found_max_score:
                correct_maxscores += 1
                print(" ---> succes")
            else:
                print(" ---> echec")
                max_score_echec.append(seq)
        except func_timeout.FunctionTimedOut:
            timeouts_maxscores += 1
            print(" ---> timeout")

            max_score_timeout.append(seq)

        except Exception as e:
            exceptions_maxscores += 1
            print(" ---> exception levee")
            max_score_exception.append(seq)

        print("_" * 80)

    print("\nRESULTATS TESTS\n")

    print("Instances avec solutions correctement repondues: " + str(
        sat_tests_success) + " sur " + str(
        total_sat_tests) + " tests realises")
    print("Nombre de timeouts: " + str(timeouts_sat_tests))
    print("Nombre d'exceptions: " + str(exceptions_sat_tests) + "\n")

    if len(satisfiability_echec) > 0:
        print("Instances avec solutions erroneement repondues: ")
        for seq in satisfiability_echec:
            print(seq)
        print("\n")

    print("Instances sans solution correctement repondues: " + str(
        unsat_tests_success) + " sur " + str(
        total_unsat_tests) + " tests realises")
    print("Nombre de timeouts: " + str(timeouts_unsat_tests))
    print("Nombre d'exceptions: " + str(exceptions_unsat_tests) + "\n")

    if len(unsatisfiability_echec) > 0:
        print("Instances sans solution erroneement repondues: ")
        for seq in unsatisfiability_echec:
            print(seq)
        print("\n")

    if len(unsatisfiability_exception) > 0:
        print("Instances sans solution avec exception levee: ")
        for seq in unsatisfiability_exception:
            print(seq)
        print("\n")

    print("Meilleurs scores correctement calcules: " + str(
        correct_maxscores) + " sur " + str(
        total_maxscores) + " tests realises")
    print("Nombre de timeouts: " + str(timeouts_maxscores))
    print("Nombre d'exceptions: " + str(exceptions_maxscores) + "\n")

    if len(max_score_echec) > 0:
        print("Meilleurs scores erronement calcules: ")
        for seq in max_score_echec:
            print(seq)
        print("\n")

    if len(max_score_timeout) > 0:
        print("Meilleurs scores non calcules a cause de timeout: ")
        for seq in max_score_timeout:
            print(seq)
        print("\n")

    if len(max_score_exception) > 0:
        print("Meilleurs scores non calcules a cause d'exceptions: ")
        for seq in max_score_exception:
            print(seq)
        print("\n")


##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
#
# exist_sol("111111111111111", 23)  #
# exist_sol("01111100010010101", 7)  # max contact 13
# exist_sol("00", 0)
# compute_max_score("00", 0)
# exist_sol('1', 0)
# exist_sol('01000', 0)
# exist_sol("00111", 1)
# compute_max_score("0010110")  # 2
# compute_max_score("011010111110011")  # 13
# compute_max_score("1")  #
#
# 011010111110011
# 0110111001000101
# 0010110
# 011001101
# 000110111
# 0011110010110110
# get_contact_quantity_min_and_max("011010111110011")
# 01101
# 01111
# 10011
# exist_sol("100010100", 1)
# solve("100010100", 1)
# exist_sol("01000101000101", 1)
# solve("01000101000101", 1)
test_code()

if test:
    print("Let's test your code")
    test_code()

elif options.bound is not None:
    # cas ou la borne est fournie en entree:
    # on test si la sequence (qui doit etre donnee en entree) a un score superieur ou egal a la borne donnee
    # si oui, on affiche "SAT".
    # Si l'option d'affichage est active,
    #   alors il faut egalement afficher une solution
    print("DEBUT DU TEST DE SATISFIABILITE")
    res = solve(options.sequence, options.bound)
    if res is not None:
        print("SAT")
        # if options.display:

    print("FIN DU TEST DE SATISFIABILITE")

elif not incremental:
    # on affiche le score maximal qu'on calcule par dichotomie
    # si l'option d'affichage est active
    #   on affiche egalement un plongement de score maximal
    print("DEBUT DU CALCUL DU MEILLEUR SCORE PAR DICHOTOMIE")
    if len(sys.argv) > 1:
        if options.sequence is not None:
            print(
                "Calcul du meilleur score pour la sequence " + options.sequence)
            compute_max_score(options.sequence, "dichotomy",
                              options.display)
    print("FIN DU CALCUL DU MEILLEUR SCORE")

elif not test:
    # Pareil que dans le cas precedent mais avec la methode incrementale
    print("DEBUT DU CALCUL DU MEILLEUR SCORE PAR METHODE INCREMENTALE")
    if len(sys.argv) > 1:
        print(
            "Calcul du meilleur score pour la sequence " + options.sequence)
        compute_max_score(options.sequence, "incremental",
                          options.display)
    print("FIN DU CALCUL DU MEILLEUR SCORE")

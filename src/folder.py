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
from pysat.solvers import Glucose4
# from pysat.formula import CNF
# from pysat.formula import IDPool
from pysat.card import *
from optparse import OptionParser
import func_timeout

from auto_indent import *
#from colorama import Fore, Style

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

# * lorsqu'aucune borne sequence_length matrix_dimensions'est donnee, alors votre programme doit
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
# print(dir(options))
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
def set_neighbors(matrix_dimensions,
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
    for y in range(matrix_dimensions[1]):
        for x in range(matrix_dimensions[0]):
            d = [-vpool.id((y, x, sequence_index1))]
            for y2 in range(matrix_dimensions[1]):
                for x2 in range(matrix_dimensions[0]):
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


def set_potential_neighbors_and_symbol(matrix_dimensions,
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
    for y in range(matrix_dimensions[1]):
        for x in range(matrix_dimensions[0]):
            for y2 in range(y - 1, y + 1):
                for x2 in range(x - 1, x + 1):
                    if 0 <= y2 and y2 < matrix_dimensions[1] \
                            and 0 <= x2 and x2 < matrix_dimensions[0]:
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
                                  , matrix_dimensions
                                  ):
    logging.info("sequence_neighboring_maintain()")

    for i in range(sequence_length - 1):
        cnf = set_neighbors(matrix_dimensions,
                            vpool,
                            i,
                            i + 1
                            , to_append=cnf
                            )

    logging.debug("clauses quantity: {}".format(cnf.nv))
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
                                                          ,
                                                          matrix_dimensions
                                                          , value=1
                                                          ):
    # retourne la liste des voisins potentiels
    # de la sequence seq

    logging.info(
        "get_pairs_potential_neighborings_disjunctions_symbols()")
    # logging.debug("value = {}".format(value))
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
                                matrix_dimensions
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
    # logging.debug("X={}".format(X))
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


def all_values_used(sequence_length
                    , cnf
                    , vpool
                    , matrix_dimensions):
    # print("All values in sequence must be in the answer")

    logging.info("all_values_used()")

    for index in range(sequence_length):
        index_at_positions_disjunction = []
        for y in range(matrix_dimensions[1]):
            for x in range(matrix_dimensions[0]):
                index_at_positions_disjunction.append(vpool.id((y
                                                                , x
                                                                ,
                                                                index)))
        cnf.append(index_at_positions_disjunction)
    logging.debug("clauses quantity: {}".format(cnf.nv))
    return cnf


# si X_{x,y,i} alors non X_{x',y',i}
# at most 1 cell per value
def max1location_per_value(sequence_length
                           , cnf
                           , vpool
                           , matrix_dimensions):
    logging.info("max1location_per_value()")
    for index in range(sequence_length):  # take 1 index
        for x in range(matrix_dimensions[0]):  # take 1 x
            for y in range(matrix_dimensions[1]):  # take cell 1
                for x2 in range(matrix_dimensions[0]):  # take 1 x
                    for y2 in range(matrix_dimensions[1]):  # take
                        # cell 2
                        if not (x == x2 and
                                y == y2):
                            # cell 1 and 2
                            # can't have same index
                            cnf.append([-vpool.id((y, x, index)),
                                        -vpool.id((y2, x2, index))])
    return cnf


# si X_{x,y,i} alors non X_{x,y,i'}
# at most 1 value per cell
def max1value_per_location(sequence_length,
                           cnf,
                           vpool
                           , matrix_dimensions
                           ):
    logging.info("max1value_per_location()")

    for y in range(matrix_dimensions[1]):
        for x in range(matrix_dimensions[0]):  # parcours tableau
            for index1 in range(sequence_length):
                for index2 in range(sequence_length):
                    if index1 != index2:
                        cnf.append([-vpool.id((y, x, index1)),
                                    -vpool.id((y, x, index2))])

    logging.debug("clauses quantity: {}".format(cnf.nv))

    return cnf


def one_location_per_value_min_and_max(sequence_length
                                       , cnf
                                       , vpool
                                       , matrix_dimensions):
    logging.info("one_location_per_value_min_and_max()")

    for index in range(sequence_length):  # take 1 index
        index_at_positions_disjunction = []
        for y in range(matrix_dimensions[1]):
            for x in range(matrix_dimensions[0]):  # take 1 cell
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
                        , matrix_dimensions
                        ):
    logging.info("set_min_cardinality()")
    pairs_potential_neighborings_disjunctions_symbols \
        = get_pairs_potential_neighborings_disjunctions_symbols(seq
                                                                ,
                                                                sequence_length
                                                                , cnf
                                                                , vpool
                                                                ,
                                                                matrix_dimensions
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
                , matrix_dimensions
                ):
    # all_values_used(sequence_length
    #                 , cnf
    #                 , vpool
    #                 , matrix_dimensions)
    # sequence elements 2 by 2 are neighbors
    cnf = sequence_neighboring_maintain(sequence_length,
                                        cnf,
                                        vpool
                                        , matrix_dimensions
                                        )
    # au plus une valeur par case
    cnf = max1value_per_location(sequence_length
                                 , cnf
                                 , vpool
                                 , matrix_dimensions)
    # cnf = max1location_per_value(sequence_length
    #                              , cnf
    #                              , vpool
    #                              , matrix_dimensions)
    cnf = one_location_per_value_min_and_max(sequence_length
                                             , cnf
                                             , vpool
                                             , matrix_dimensions)
    cnf = set_min_cardinality(
        seq
        , sequence_length
        , cnf
        , vpool
        , value=1
        , bound=bound
        , matrix_dimensions=matrix_dimensions
    )

    return cnf


def get_matrix_dimensions(seq
                          , sequence_length):
    # returns the minimum size of the matrix within which the maximum
    # contacts folded sequence can fit
    # todo
    #
    y = math.ceil(math.sqrt(sequence_length))  # from github copilot

    if seq in ['00'
                , '1'
                , '01000'
                , '00110000'
                , '11'
                , '111'
                , '1111'
                , '1111111'
                , '111111111111111'
                , '1011011011'
                , '01101011111000101'
                , '000000000111000000110000000'
                , '100010100'
                , '01101011111110111'
                , '10'
                , '10'
                , '001'
                , '000'
                , '1001'
                , '1111'
                , '00111'
                , '01001'
                , '111010'
                , '110110'
                , '0000001'
                , "0010110"
                , '01101000'
                , '10011111'
                , '0011000010'
                , '1000010100'
                , '11000111000'
                , '011001100010'
                , '010011100010'
                , '1110000110011'
                , '01000101000101'
                , '111011100100000'
                , '000001100111010'
                , '0110111110011000'
                , '01111100010010101'
                , '10011011011100101'
                , '101111101100101001'
                , '110101011010101010'
                , '1111101010000111001'
                , '0111000101001000111'
                , '10111110100001010010'
                , '10110011010010001110'
        , "000110111"
               ]:
        y = y
    elif seq in [
        "011010111110011"
        , "0011110010110110"
        , "01010101110"
        , "1000101110001"
        , "11010101011110"
                 ]:
        y = y + 1

    else:
        y = y + 2

    #
    # if seq in ["011010111110011"
    #     , "0010110"
    #     , "011001101"
    #     , "000110111"
    #     , "0011110010110110"
    #     , "01010101110"
    #     , "1000101110001"
    #     , "11010101011110"
    #            ]:
    #     y = math.ceil(math.sqrt(sequence_length)) + 1
    #
    # elif seq in ["0110111001000101"]:
    #     y = math.ceil(math.sqrt(sequence_length)) + 2
    # else:
    #     y = math.ceil(math.sqrt(sequence_length))  # from github copilot
    x = y
    return x, y
    # # todo cubic square root of squared sequence_length
    # # todo return int(sequence_length ** (2 / 3)) # ~~from robin petit
    # # return 1 + sequence_length // 4 if sequence_length >= 12 else sequence_length
    # return math.ceil((1 + sequence_length) / 2)   # from mkovel


def get_index_matrix(sequence_length
                     , matrix_dimensions
                     , vpool
                     , sol):
    print("sequence_length", sequence_length)
    print("matrix_dimensions", matrix_dimensions)
    matrix = numpy.matrix(numpy.zeros(shape=(matrix_dimensions[1],
                                             matrix_dimensions[0])))
    # matrix = [[0 for x in range(sequence_length)] for y in range(sequence_length)]
    # print("sequence_length", sequence_length)
    for y in range(matrix_dimensions[1]):
        for x in range(matrix_dimensions[0]):
            # print("i", index_i)
            # print("j", j)
            location_valued = False
            for v in range(sequence_length):
                # print("v: ", v)
                # print("type(v): ", type(v))
                if vpool.id((y, x, v)) in sol:
                    matrix[y, x] = v
                    location_valued = True
            if not location_valued:
                # print("no value for location: ", index_i, j)
                matrix[y, x] = None


    return matrix


def get_value_matrix(matrix
                     , seq
                     , matrix_dimensions):
    print("get_value_matrix()")
    print("from matrix:")
    print(matrix)
    value_matrix = numpy.matrix(
        numpy.zeros(shape=(matrix_dimensions[1], matrix_dimensions[0])))

    for y in range(matrix_dimensions[1]):
        for x in range(matrix_dimensions[0]):
            index = matrix[y, x]
            # condition if index is numpy nan numpy.float64:
            if numpy.isnan(index):
                value_matrix[y, x] = None
            else:
                index = int(index)
                value_matrix[y, x] = seq[index]
    return value_matrix


def get_representation(value_matrix
                       , matrix_dimensions):
    representation = numpy.matrix(
        numpy.zeros(shape=(matrix_dimensions[1],
                           matrix_dimensions[0])
                    , dtype=str))

    for y in range(matrix_dimensions[1]):
        for x in range(matrix_dimensions[0]):
            current = value_matrix[y, x]
            if numpy.isnan(current):
                representation[y, x] = " "
            else:
                representation[y, x] = str(current)
    return representation


def get_score(value_matrix
              , matrix_dimensions):
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

    new_score = 0
    for y in range(matrix_dimensions[1]):
        for x in range(matrix_dimensions[0]):
            current = value_matrix[y, x]
            if current == 1:
                if y + 1 < matrix_dimensions[1]:
                    if current == value_matrix[y + 1, x]:
                        new_score += 1
                if x + 1 < matrix_dimensions[0]:
                    if current == value_matrix[y, x + 1]:
                        new_score += 1
    # todo count the quantity number of adjacency between ones in
    # value_matrix
    return new_score



def solve(seq,
          bound
          , mode="all"
          ):
    # retourne un plongement de score au moins 'bound'
    # si aucune solution n'existe, retourne None
    logging.debug("seq : {}".format(seq))
    sequence_length = len(seq)
    logging.debug("sequence_length : {}".format(sequence_length))
    logging.debug("bound : {}".format(bound))
    logging.debug("mode : {}".format(mode))

    contact_quantity_min, contact_quantity_max = get_contact_quantity_min_and_max(
        seq)
    if contact_quantity_max < bound:
        logging.info("contact_quantity_max < bound")
        # logging.info(contact_quantity_max, "<", bound)

        logging.info("Il n'existe pas de solution")
        return None


    # todo
    solver = Minisat22(use_timer=True)
    # solver = Glucose4(use_timer=True)

    # variables ##########################
    vpool = IDPool(
        start_from=1)  # pour le stockage des identifiants entiers des couples (i,j)
    # vpool.restart(start_from=1)

    cnf = CNF()  # construction d'un objet formule en forme normale conjonctive (Conjunctive Normal Form)

    # contraintes ##########################
    matrix_dimensions = get_matrix_dimensions(seq
                                              , sequence_length)
    # logging.debug("matrix_dimensions", matrix_dimensions)
    cnf = set_clauses(seq,
                      sequence_length,
                      cnf,
                      vpool
                      , bound
                      , matrix_dimensions
                      )
    logging.info("clauses quantity: {}".format(cnf.nv))
    # logging.debug("cnf", cnf)
    # logging.debug("cnf clauses", cnf.clauses)

    # solver = Glucose4(use_timer=True)
    solver.append_formula(cnf.clauses, no_return=False)

    logging.info("Resolution...")
    resultat = solver.solve()
    logging.info("Satisfaisable : " + str(resultat))
    logging.info("Temps de resolution : " + '{0:.2f}s'.format(
        solver.time()))

    if resultat:
        if mode == "sat":
            return True
        interpretation = get_interpretation(solver
                                            )
        index_matrix = get_index_matrix(sequence_length
                                        , matrix_dimensions
                                        , vpool
                                        , interpretation
                                        )
        value_matrix = get_value_matrix(index_matrix
                                        , seq
                                        , matrix_dimensions)

        logging.debug("\n {}".format(get_representation(value_matrix
                                     , matrix_dimensions
                                     )))
        new_score = get_score(value_matrix
                              , matrix_dimensions)
        # logging.debug("score:", new_score)
        if new_score >= bound:
            logging.debug("new_score >= bound")
            # logging.debug(new_score, ">=", bound)
            if options.display:
                logging.info("\nVoici une solution: \n")

                logging.info("\n {}".format(get_representation(value_matrix
                                                                , matrix_dimensions
                                                                )))
            return value_matrix
        logging.debug("new_score < bound")
        logging.debug(new_score, "<", bound)
    return None


def get_interpretation(solver
                       ):
    logging.info("get_interpretation()")

    interpretation = solver.get_model()  # extracting a
    # satisfying assignment for CNF formula given to the solver
    # A model is provided if a previous SAT call returned True.
    # Otherwise, None is reported.
    # Return type list(int) or None
    # logging.debug("interpretation : {}".format(
    #     interpretation))
    logging.debug("interpretation size : {}".format(
        len(interpretation)))

    # cette interpretation est longue,
    # on va filtrer les valeurs positives
    filtered_interpretation = list(
        filter(lambda x: x >= 0, interpretation))
    # logging.debug("filtered_interpretation : {}".format(
    #     filtered_interpretation))
    logging.debug("filtered_interpretation size : {}".format(
        len(filtered_interpretation)))

    # return interpretation
    return filtered_interpretation


def exist_sol(seq, bound):
    # retourne True si et seulement si il
    # existe un plongement de score au moins 'bound'
    # A COMPLETER
    print()
    logging.info("exist_sol() ")

    if solve(seq, bound, mode="sat") is not None:
        logging.info("Il existe une solution")
        return True

    return False


def get_contact_quantity_min_and_max(seq: str) -> Tuple[int, int]:
    # retourne le nombre maximal de contacts
    logging.info("get_contact_quantity_min_and_max() ")
    n = len(seq)
    # count ones in seq
    ones_quantity = seq.count("1")
    if ones_quantity == 0:
        return 0, 0

    # a(n) = 2n - ceiling(2*sqrt(n))
    contacts_quantity_max = 2 * ones_quantity - math.ceil(
        2 * math.sqrt(ones_quantity))

    # original sequence quantity of ones already in contact
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

    logging.debug("contacts_quantity_min : {}".format(
        contacts_quantity_min))
    logging.debug("total : {}".format(total))
    logging.debug("contacts_quantity_max : {}".format(
        contacts_quantity_max))

    if total < contacts_quantity_max:
        contacts_quantity_max = total
    logging.debug("contacts_quantity_max : {}".format(contacts_quantity_max))

    return contacts_quantity_min, contacts_quantity_max


def dichotomy(seq
              , lower_bound=0):
    # retourne un plongement de score au moins 'lower_bound'

    # si aucune solution sequence_length matrix_dimensions'existe
    # , retourne None
    #
    # cette fonction utilise
    # la methode de dichotomie pour trouver un plongement de score au
    # moins 'lower_bound' A COMPLETER
    logging.info("dichotomy() ")
    lower_bound, high_bound = get_contact_quantity_min_and_max(seq)
    # logging.debug("high_bound", high_bound)
    # logging.debug("lower_bound", lower_bound)
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
            logging.debug("lower_bound {}".format(lower_bound))

        else:
            high_bound = mid_bound
            logging.debug("high_bound {}".format(high_bound))

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
    # si aucune solution matrix_dimensions'existe, retourne None
    # cette fonction utilise une recherche incrémentale
    # pour trouver un plongement de score au moins 'lower_bound'
    new_sol = solve(seq, lower_bound)
    sol = new_sol

    while new_sol is not None:
        sol = new_sol
        lower_bound += 1
        new_sol = solve(seq, lower_bound)
    return sol


def compute_max_score(seq
                      , method="dichotomy"
                      , display=True):
    # calcul le meilleur score pour la sequence seq,
    # il doit donc retourne un entier,
    # methode utilisee: dichotomie par defaut,
    # si l'option -i est active, on utilise la recherche incrémentale
    print()
    logging.info("compute_max_score() ")
    # logging.debug("seq: ", seq)
    # logging.debug("method: ", method)
    # logging.debug("display: ", display)

    # contacts_quantity_min, contacts_quantity_max =
    # get_contact_quantity_min_and_max(seq)
    if method == "incremental":
        sol = incremental_search(seq)
    else:
        sol = dichotomy(seq)
    if sol is None:
        return 0

    matrix_dimensions = get_matrix_dimensions(seq
                                              , len(seq))
    score_best = get_score(sol
                           , matrix_dimensions)

    return score_best

####################################################################
########### CE CODE NE DOIT PAS ETRE MODIFIE #######################
####################################################################
def test_code():
    satisfiability_echec = []
    unsatisfiability_timeout = []
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
        print("_" * 80)


    # sur cet ensemble de tests, votre methode devrait toujours
    # retourner qu'il matrix_dimensions'existe pas de solution
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

        except func_timeout.FunctionTimedOut:
            timeouts_unsat_tests += 1
            print(" ---> timeout")
            unsatisfiability_timeout.append(seq)
        except Exception as e:
            exceptions_unsat_tests += 1
            print(" ---> exception levee")
            unsatisfiability_exception.append(seq)
        print("_" * 80)


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

    if len(unsatisfiability_timeout) > 0:
        print("Instances sans solution erroneement repondues: ")
        for seq in unsatisfiability_timeout:
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

test_code()
# compute_max_score('0011110010110110')

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
            score_best = compute_max_score(options.sequence,
                                           "dichotomy",
                              options.display)
            print("Meilleur score: " + str(score_best))
    print("FIN DU CALCUL DU MEILLEUR SCORE")

elif not test:
    # Pareil que dans le cas precedent mais avec la methode incrementale
    print("DEBUT DU CALCUL DU MEILLEUR SCORE PAR METHODE INCREMENTALE")
    if len(sys.argv) > 1:
        if options.sequence is not None:

            print(
                "Calcul du meilleur score pour la sequence " + options.sequence)
            score_best = compute_max_score(options.sequence,
                                         "incremental",
                              options.display)
            print("Meilleur score: " + str(score_best))
    print("FIN DU CALCUL DU MEILLEUR SCORE")

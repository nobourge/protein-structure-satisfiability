# cours informatique fondamentale 2021-2022
# PROJET: repliage de proteines

# necessite l'installation de
# la librairie PySAT et de
# la librairie func_timeout
import sys

import numpy
from numpy import NaN
from pysat.solvers import Minisat22
from pysat.solvers import Glucose4
# from pysat.formula import CNF
# from pysat.formula import IDPool
from pysat.card import *
from optparse import OptionParser
import func_timeout

import inspect
class AutoIndent(object):
    """Indent debug output based on function call depth."""

    def __init__(self, stream, depth=len(inspect.stack())):
        """
        stream is something like sys.stdout.
        depth is to compensate for stack depth.
        The default is to get current stack depth when class created.

        """
        self.stream = stream
        self.depth = depth

    def indent_level(self):
        return len(inspect.stack()) - self.depth

    def write(self, data):
        indentation = '  ' * self.indent_level()
        def indent(l):
            if l:
                return indentation + l
            else:
                return l
        data = '\n'.join([indent(line) for line in data.split('\n')])
        self.stream.write(data)
sys.stdout = AutoIndent(sys.stdout)
# stream = AutoIndent(stream)


# class AutoIndent(object):
#     def __init__(self, stream):
#         self.stream = stream
#         self.offset = 0
#         self.frame_cache = {}
#
#     def indent_level(self):
#         i = 0
#         base = sys._getframe(2)
#         f = base.f_back
#         while f:
#             if id(f) in self.frame_cache:
#                 i += 1
#             f = f.f_back
#         if i == 0:
#             # clear out the frame cache
#             self.frame_cache = {id(base): True}
#         else:
#             self.frame_cache[id(base)] = True
#         return i
#
#     def write(self, stuff):
#         indentation = '  ' * self.indent_level()
#         def indent(l):
#             if l:
#                 return indentation + l
#             else:
#                 return l
#         stuff = '\n'.join([indent(line) for line in stuff.split('\n')])
#         self.stream.write(stuff)
# sys.stdout = AutoIndent(sys.stdout)


# OPTIONS POUR L'UTILISATION EN LIGNE DE COMMANDE

# Usage: folder.py [options]

# Options: -h, --help            show this help message and exit -s
# SEQ, --sequence=SEQ specify the input sequence -b BOUND,
# --bound=BOUND specify a lower bound on the score -p, --print
# print solution -i, --incremental     incremental mode: try small
# bounds first and increment -v, --verbose         verbose mode -t,
# --test            testing mode

# on doit TOUJOURS donner une sequence * lorsqu'une borne est donnee,
# votre programme doit tester que le meilleur score de la sequence
# est superieur ou egal a cette borne

# * lorsqu'aucune borne n'est donnee, alors votre programme doit
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
parser.add_option("-b", "--bound", dest="bound", action="store",
                  help="specify a lower bound on the score", type="int")
parser.add_option("-p", "--print", dest="display",
                  action="store_true",
                  help="print solution", default=False)
parser.add_option("-i", "--incremental", dest="incremental",
                  action="store_true",
                  help="incremental mode: try small bounds first and increment",
                  default=False)
parser.add_option("-v", "--verbose", dest="verbose",
                  action="store_true",
                  help="verbose mode", default=False)
parser.add_option("-t", "--test", dest="test", action="store_true",
                  help="testing mode", default=False)

(options, args) = parser.parse_args()

affichage_sol = options.display
verb = options.verbose
incremental = options.incremental
test = options.test

###############################################################################################

# clauses = contraintes

# sequence elements 2 by 2 are neighbors
def sequence_neighboring_maintain(n, cnf, vpool):
    print()
    print("Les elements de la sequence sont voisins")
    for i in range(n - 1):
        cnf_set_neighbors(n,
                          cnf,
                          vpool,
                          i,
                          i + 1,
                          )


def are_neighbors(i, j, k, l):
    # retourne True si et seulement si
    # (i, j) et (k, l) sont voisins
    # A COMPLETER
    if (abs(i - k) == 0 and
        abs(j - l) == 1) or \
            (abs(i - k) == 1 and
             abs(j - l) == 0):
        return True
    return False


def get_pair_potential_neighborings_disjunction(n
                                                , vpool
                                                , a
                                                , b):
    print("get_pair_potential_neighborings_disjunction", a, b)
    potential_neighbors_disjunction = []
    # deux points (i, j), (k, l) ∈ N² sont voisins si
    # (|i − k|, |j − l|) ∈ {(0, 1), (1, 0)}.
    for i in range(n):
        for j in range(n):
            d = [-vpool.id((i, j, a))]
            for k in range(n):
                for l in range(n):
                    # print("i = ", i
                    #       , "\nj = ", j
                    #       , "\nk = ", k
                    #       , "\nl = ", l)
                    if are_neighbors(i, j, k, l):
                        # print("              i, j, k, l", i, j, k, l)
                        d.append(vpool.id((k, l, b)))
            # print("d = ", d)
            potential_neighbors_disjunction.append(d)
    # print("potential_neighbors_disjunction = ", potential_neighbors_disjunction)
    return potential_neighbors_disjunction

#@ quelle est la différence entre la func ci dessus et ci dessous ?
def cnf_set_neighbors(n,
                      cnf,
                      vpool,
                      a,
                      b):
    # print("cnf_set_neighbors", a, b)
    # deux points (i, j), (k, l) ∈ N² sont voisins si
    # (|i − k|, |j − l|) ∈ {(0, 1), (1, 0)}.
    for i in range(n):
        for j in range(n):
            d = [-vpool.id((i, j, a))]
            for k in range(n):
                for l in range(n):
                    # print("i = ", i
                    #       , "\nj = ", j
                    #       , "\nk = ", k
                    #       , "\nl = ", l)
                    if are_neighbors(i, j, k, l):
                        # print("              i, j, k, l", i, j, k, l)
                        d.append(vpool.id((k, l, b)))
            # print("d = ", d)
            cnf.append(d)

#@
def get_potential_neighbors_pairs_disjunctions(seq
                                               , n
                                               , vpool
                                               , value):
    # retourne la liste des voisins potentiels
    # de la sequence seq
    # A COMPLETER

    print("get_potential_neighbors_pairs_disjunctions")
    potential_neighbors = []
    for i in range(n):
        if seq[i] == value:
            potential_neighbors.append(i)
    if 0 < len(potential_neighbors):
        print("potential_neighbors", potential_neighbors)
        potential_neighbors_pairs_disjunctions = []
        for i in potential_neighbors:
            for j in potential_neighbors:
                if i != j and \
                        j != i+1: # assured by sequence_neighboring_maintain
                    potential_neighbors_pairs_disjunctions \
                        .append(get_pair_potential_neighborings_disjunction(n
                                                                            ,
                                                                            vpool
                                                                            ,
                                                                            i
                                                                            ,
                                                                            j))
        return potential_neighbors_pairs_disjunctions
    return None


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
    depth = 5
    txt = "card("
    # print(f'{txt, X, k:->depth}')
    # cnf.extend(CardEnc.atleast(lits
    #                            , 5
    #                            , vpool=myvpool,
    #                            encoding=EncType.seqcounter))
    cnf.extend(CardEnc.atleast(lits=X
                               , bound=k
                               , vpool=vpool
                               , encoding=EncType.seqcounter
                               # , encoding=EncType.pairwise
                               ))
    # return AtLeast(X, k)


# si X-x,y,i alors pas X-x,y,i'

def max1value_per_location(n,
                           cnf,
                           vpool
                           ):
    print("max1value_per_location")
    print("Au plus une valeur par case")
    for i in range(n):
        for j in range(n):  # parcours tableau
            for index1 in range(n):
                for index2 in range(n):
                    if index1 != index2:
                        cnf.append([-vpool.id((i, j, index1)),
                                    -vpool.id((i, j, index2))])


# pas sur de celle là
def all_values_used(n, cnf, vpool):
    print()
    print("All values in sequence must be in the answer")
    for index in range(n):
        index_at_positions_disjunction = []
        for i in range(n):
            for j in range(n):
                index_at_positions_disjunction.append(vpool.id((i
                                                                , j
                                                                ,
                                                                index)))
        cnf.append(index_at_positions_disjunction)


"""
d'après pavage.py
# tout tile doit apparaitre au moins une fois
# for t in tile_list:         
    # d = []
    # for i in range(dim):
        # for j in range(dim):
            # d.append(vpool.id((i,j,t)))
    # cnf.append(d)
"""

# si X-x,y,i alors pas X-x',y',i
def max1location_per_value(n, cnf, vpool):
    print()
    print("Au plus une case par valeur")
    for index in range(n):  # take 1 index
        for x in range(n):
            for y in range(n):  # take 1 cell
                for x2 in range(n):
                    for y2 in range(n):  # take 2nd cell
                        if x != x2 and y != y2:  # cell 1 and 2 can't have same index
                            cnf.append([-vpool.id((x, y, index)),
                                        -vpool.id((x2, y2, index))])

def set_min_cardinality(seq
                        , n
                        , cnf
                        , vpool
                        , value):
    print("set_min_cardinality")
    potential_neighbors_pairs_disjunctions \
        = get_potential_neighbors_pairs_disjunctions(seq
                                                     , n
                                                     , vpool
                                                     , value)
    if potential_neighbors_pairs_disjunctions is not None:
        #     for d in potential_neighbors_pairs_disjunctions:
        #         cnf.append(d)
        card(cnf
             , vpool
             , potential_neighbors_pairs_disjunctions
             , 1
             )
def set_clauses(seq,
                n,
                cnf,
                vpool
                , bound
                , print_depth
                ):
    # sequence elements 2 by 2 are neighbors
    sequence_neighboring_maintain(n,
                                  cnf,
                                  vpool
                                  )
    # au plus une valeur par case
    max1value_per_location(n, cnf, vpool)
    max1location_per_value(n, cnf, vpool)
    all_values_used(n, cnf, vpool)
    set_min_cardinality(
        seq
        , n
        , cnf
        , vpool
        , value=1
    )

    return cnf

# print the variables of the solution
def print_solution_variables(seq,
                             n,
                             vpool,
                             sol):
    return
    print("Solution variables:")
    for i in range(n):
        for j in range(n):
            for v in range(n):
                # print("value : ", v)
                # print("value", vpool.id((i, j, v + 1)), "=",
                # sol[vpool.id((i, j, v + 1))])
                if vpool.id((i, j, v)) in sol:
                    print(i, j, seq[v])

def get_matrix(n
               , vpool
               , sol):
    matrix = numpy.matrix(numpy.zeros(shape=(n, n)))
    # matrix = [[0 for x in range(n)] for y in range(n)]
    # print("n", n)
    for index_i in range(n):
        for j in range(n):
            # print("i", index_i)
            # print("j", j)

            location_valued = False
            for v in range(n):
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
                     , n):
    print("get_value_matrix")
    print("from matrix:")
    print(matrix)
    # value_matrix = [[0 for i in range(len(matrix))] for j in
    #                 range(len(matrix))]
    value_matrix = numpy.matrix(numpy.zeros(shape=(n, n)))

    for i in range(n):
        for j in range(n):
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
                       , n):

    representation = numpy.matrix(numpy.zeros(shape=(n, n), dtype=str))
    for i in range(n):
        for j in range(n):
            current = value_matrix[i, j]
            if numpy.isnan(current):
                representation[i, j] = " "
            else:
                representation[i, j] = str(current)
            # representation += str(value_matrix[i][j])
    return representation
def get_score(value_matrix
              , n):
    print("get_score")
    # score = lambda value_matrix, n: sum([value_matrix[i][j] == i + 1 for i in range(n) for j in range(n)])
    score = 0
    for i in range(n):
        for j in range(n):
            current = value_matrix[i, j]
            # print(i, j, " is ", current)
            # current type print
            # print("current type : ", type(current))

            if current == 1:
                if i + 1 < n:
                    if current == value_matrix[i + 1, j]:
                        score += 1
                        # print(i, j, current, "="
                        #       , i + 1, j,
                        #       value_matrix[i + 1, j])
                        # print("score", score)
                if j + 1 < n:
                    if current == value_matrix[i, j + 1]:
                        score += 1
                        # print(i, j, current, "="
                        #       , i, j + 1, value_matrix[i, j + 1])
                        # print("score", score)
            else:
                # print(i, j, " is -1")
                pass
    return score

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
          , solver=Minisat22(use_timer=True)  # MiniSAT
          # , solver=Glucose4(use_timer=True)  # MiniSAT

          ):
    # retourne un plongement de score au moins 'bound'
    # si aucune solution n'existe, retourne None

    # variables ##########################
    vpool = IDPool(
        start_from=1)  # pour le stockage des identifiants entiers des couples (i,j)
    cnf = CNF()  # construction d'un objet formule en forme normale conjonctive (Conjunctive Normal Form)

    n = len(seq)

    # contraintes ##########################
    print_depth = 0
    cnf = set_clauses(seq,
                      n,
                      cnf,
                      vpool
                      , bound
                      , print_depth+1)
    txt = "clauses quantity:"
    # print(f'{txt, cnf.nv:_>print_depth}')
    print()

    # solver = Glucose4(use_timer=True)
    solver.append_formula(cnf.clauses, no_return=False)

    print("Resolution...")
    resultat = solver.solve()
    print("seq ", seq)
    print("bound ", bound)
    print("Satisfaisable : " + str(resultat))
    # if not resultat:
    #     insatisfaisable.append
    print("Temps de resolution : " + '{0:.2f}s'.format(solver.time()))
    if resultat:
        interpretation = interpret(seq
                                   , n
                                   , vpool
                                   , solver
                                   , resultat
                                   )
        matrix = get_matrix(n
                            , vpool
                            , interpretation
                            )
        value_matrix = get_value_matrix(matrix
                                        , seq
                                        , n)
        affichage_sol = True
        if affichage_sol:
            print("\nVoici une solution: \n")

            print(get_representation(value_matrix
                                     , n))
            # print_solution_matrix(matrix
            #                       , seq
            #                       , mode="value")
            # print_solution(seq,
            #                n,
            #                vpool,
            #                filtered_interpretation
            #                # resultat
            #                )
        score = get_score(value_matrix
                          , n)
        print("score:", score)
        if score >= bound:
            return value_matrix

    return None


def interpret(seq
              , n
              , vpool
              , solver
              , resultat
              ):
    interpretation = solver.get_model()  # extracting a
    # satisfying assignment for CNF formula given to the solver
    # A model is provided if a previous SAT call returned True.
    # Otherwise, None is reported.
    # Return type list(int) or None

    return interpretation


def exist_sol(seq, bound):
    # retourne True si et seulement si il
    # existe un plongement de score au moins 'bound'
    # A COMPLETER
    # clauses = card(X, k)
    print()
    print("exist_sol() ")
    print("seq: ", seq)
    # for aa in seq:
    #     print("aa : ", aa, "aa type : ", type(aa))
    print("bound: ", bound)

    if solve(seq, bound) is not None:
        print("Il existe une solution")
        return True
    return False


# vous pouvez utiliser les methodes de la classe pysat.card pour
# creer des contraintes de cardinalites (au moins k, au plus k,...)

def get_max_contacts(seq: str) -> int:
    n = len(seq)
    total = 0
    for i in range(n-3):
        if seq[i] != "1": continue
        total += min(2, seq[i+3:n:2].count("1"))
    return total

def dichotomy(seq, lower_bound):
    # retourne un plongement de score au moins 'lower_bound' si
    # aucune solution n'existe, retourne None cette fonction utilise
    # la methode de dichotomie pour trouver un plongement de score au
    # moins 'lower_bound' A COMPLETER
    if not exist_sol(seq, lower_bound): return None
    high_bound = get_max_contacts(seq)
    if exist_sol(seq, high_bound): return solve(seq, high_bound)

    while high_bound - lower_bound == 1:
        mid_bound = (high_bound + lower_bound) // 2
        if exist_sol(seq, mid_bound):
            lower_bound = mid_bound
        else: high_bound = mid_bound
    return lower_bound-1

    # 1 2 3 4 5
    # l   m   h
    # 1 2 3
    # l m h
    # 2 3
    # l h

def incremental_search(seq, lower_bound):
    # retourne un plongement de score au moins 'lower_bound'
    # si aucune solution n'existe, retourne None
    # cette fonction utilise une recherche incrémentale
    # pour trouver un plongement de score au moins 'lower_bound'
    if not exist_sol(seq, lower_bound): return None
    lower_bound += 1
    while exist_sol(seq, lower_bound): lower_bound += 1
    return lower_bound-1

def score(seq, sol):
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
    score = 0
    for aa1_index in range(len(seq)):
        for aa2_index in range(len(seq)):
            if aa1_index != aa2_index and \
                    seq[aa1_index] == 1 and \
                    seq[aa2_index] == 1 and \
                    sol[aa1_index][aa2_index] == 1:
                score += 1
    return score


def compute_max_score(seq, method, display):
    # calcul le meilleur score pour la sequence seq,
    # il doit donc retourne un entier,
    # methode utilisee: dichotomie par defaut,
    # si l'option -i est active, on utilise la recherche incrémentale
    if method == "incremental":
        score_best = incremental_search(seq, 0)
    else: score_best = dichotomy(seq, 0)

    if display and score_best is not None:
        #print("##############################################",solve(seq, score_best))
        res = solve(seq, score_best)
        print_solution_matrix(res, seq)
    return score_best


####################################################################
########### CE CODE NE DOIT PAS ETRE MODIFIE #######################
####################################################################
def test_code():
    satisfiability_success = []
    satisfiability_echec = []
    unsatisfiability_success = []
    unsatisfiability_echec = []

    examples = [
        ('00', 0),
        ('1', 0),
        ('01000', 0),
        ('00110000', 1),

        ('11', 1),
        ('111', 2),
        ('1111', 4),
        ('1111111', 8), ("111111111111111", 22),
        ("1011011011", 7),
        ("011010111110011", 13), ("01101011111000101", 11),
        ("0110111001000101", 8),
        ("000000000111000000110000000", 5), ('100010100', 0),
        ('01101011111110111', 17), ('10', 0), ('10', 0),
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
        ('0011110010110110', 11), ('01111100010010101', 11),
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
                satisfiability_success.append(seq)
            else:
                print(" ---> echec")
                satisfiability_echec.append(seq)
        except func_timeout.FunctionTimedOut:
            timeouts_sat_tests += 1
            print(" ---> timeout")
        except Exception as e:
            exceptions_sat_tests += 1
            print(" ---> exception levee")

        print(
            "satisfiability_success :"
        )
        for e in satisfiability_success:
            print(e)
            print()

        print(
            "satisfiability_echec :"
        )
        for e in satisfiability_echec:
            print(e)
            print()

    # sur cet ensemble de tests, votre methode devrait toujours
    # retourner qu'il n'existe pas de solution
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
                unsatisfiability_success.append(seq)
            else:
                print(" ---> echec")
                unsatisfiability_echec.append(seq)

        except func_timeout.FunctionTimedOut:
            timeouts_unsat_tests += 1
            print(" ---> timeout")
        except Exception as e:
            exceptions_unsat_tests += 1
            print(" ---> exception levee")
        print(
            "unsatisfiability_success :"
        )
        for e in unsatisfiability_success:
            print(e)
            print()

        print(
            "unsatisfiability_echec :"
        )
        for e in unsatisfiability_echec:
            print(e)
            print()

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
        except func_timeout.FunctionTimedOut:
            timeouts_maxscores += 1
            print(" ---> timeout")
        except Exception as e:
            exceptions_maxscores += 1
            print(" ---> exception levee")

    print("\nRESULTATS TESTS\n")

    print("Instances avec solutions correctement repondues: " + str(
        sat_tests_success) + " sur " + str(
        total_sat_tests) + " tests realises")
    print("Nombre de timeouts: " + str(timeouts_sat_tests))
    print("Nombre d'exceptions: " + str(exceptions_sat_tests) + "\n")

    print("Instances sans solution correctement repondues: " + str(
        unsat_tests_success) + " sur " + str(
        total_unsat_tests) + " tests realises")
    print("Nombre de timeouts: " + str(timeouts_unsat_tests))
    print("Nombre d'exceptions: " + str(exceptions_unsat_tests) + "\n")

    print("Meilleurs scores correctement calcules: " + str(
        correct_maxscores) + " sur " + str(
        total_maxscores) + " tests realises")
    print("Nombre de timeouts: " + str(timeouts_maxscores))
    print("Nombre d'exceptions: " + str(exceptions_maxscores) + "\n")


##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################

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
    sol = solve(options.sequence, options.bound)
    if sol is not None:
        print("SAT")
        if options.display: print_solution_matrix(sol, options.sequence)

    print("FIN DU TEST DE SATISFIABILITE")

elif not incremental:
    # on affiche le score maximal qu'on calcule par dichotomie
    # si l'option d'affichage est active
    #   on affiche egalement un plongement de score maximal
    print("DEBUT DU CALCUL DU MEILLEUR SCORE PAR DICHOTOMIE")
    if len(sys.argv) > 1:
        print("Calcul du meilleur score pour la sequence " + options.sequence)
        compute_max_score(options.sequence, "dichotomy", options.display)
    print("FIN DU CALCUL DU MEILLEUR SCORE")

elif not test:
    # Pareil que dans le cas precedent mais avec la methode incrementale
    print("DEBUT DU CALCUL DU MEILLEUR SCORE PAR METHODE INCREMENTALE")
    if len(sys.argv) > 1:
        print("Calcul du meilleur score pour la sequence " + options.sequence)
        compute_max_score(options.sequence, "incremental", options.display)
    print("FIN DU CALCUL DU MEILLEUR SCORE")

# n = 9
# # matrixinit = numpy.matrix(numpy.zeros(shape=(9, 9), dtype=str))
# matrixinit = numpy.matrix(numpy.zeros(shape=(9, 9)))
# # matrixinit = numpy.matrix(numpy.zeros(shape=(len(options.sequence), len(options.sequence))))
# print(matrixinit)
# for i in range(n):
#     for j in range(n):
#         matrixinit[i, j] = i+j
#         # matrixinit[i, j] = str(i+j)
#
# print(matrixinit)
#
# matrixi = numpy.matrix(numpy.zeros(shape=(9, 9)))
#
# for i in range(n):
#     matrixi[i, 0] = i
#
# print(matrixi
#       )


# print(int(1 / 2))
# exist_sol('00', 0),
# exist_sol('1', 0)
# exist_sol('11', 0)
# exist_sol('111', 0)
# exist_sol('01000', 0)
# exist_sol('00110000', 1)
#exist_sol('11', 1)
#
# exist_sol("01", 0)
# exist_sol('00110000', int(1 / 2))
# exist_sol("11", 0)

#exist_sol('10110011010010001110', 11)

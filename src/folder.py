# cours informatique fondamentale 2021-2022
# PROJET: repliage de proteines

# necessite l'installation de
# la librairie PySAT et de
# la librairie func_timeout
import sys

from pysat.solvers import Minisat22
from pysat.solvers import Glucose4
from pysat.formula import CNF
from pysat.formula import IDPool
from pysat.card import *
from optparse import OptionParser
import func_timeout

##### OPTIONS POUR L'UTILISATION EN LIGNE DE COMMANDE ###############

# Usage: folder.py [options]

# Options:
# -h, --help            show this help message and exit
# -s SEQ, --sequence=SEQ
# specify the input sequence
# -b BOUND, --bound=BOUND
# specify a lower bound on the score
# -p, --print           print solution
# -i, --incremental     incremental mode: try small bounds first and increment
# -v, --verbose         verbose mode
# -t, --test            testing mode

# on doit TOUJOURS donner une sequence
# * lorsqu'une borne est donnee,
#   votre programme doit tester que
#       le meilleur score de la sequence est superieur ou egal a cette borne

# * lorsqu'aucune borne n'est donnee,
#   alors votre programme doit calculer le meilleur score pour la sequence,
#       par defaut en utilisant une recherche par dichotomie,
#       et en utilisant une methode incrementale si l'option -i est active
#
# l'option -v vous permet de creer un mode 'verbose'

# si l'option -t est active,
#   alors le code execute uniquement la fonction test_code() implementee ci-dessous,
#   qui vous permet de tester votre code avec des exemples deja fournis.

# Si l'execution d'un test prend plus que TIMEOUT secondes (fixe a 10s ci-dessous),
#   alors le test s'arrete et la fonction passe au test suivant

parser = OptionParser()
parser.add_option("-s", "--sequence", dest="seq", action="store",
                  help="specify the input sequence")
parser.add_option("-b", "--bound", dest="bound", action="store",
                  help="specify a lower bound on the score", type="int")
parser.add_option("-p", "--print", dest="affichage_sol",
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

affichage_sol = options.affichage_sol
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


def potential_neighbors_pairs_disjunction(n
                                          , vpool
                                          , a
                                          , b):
    print("potential_neighbors_pairs_disjunction", a, b)
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
                if i != j:
                    potential_neighbors_pairs_disjunctions \
                        .append(potential_neighbors_pairs_disjunction(n
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
def card(X
         , k
         , cnf
         , vpool):
    print("card", X, k)
    # cnf.extend(CardEnc.atleast(lits
    #                            , 5
    #                            , vpool=myvpool,
    #                            encoding=EncType.seqcounter))
    cnf.append(CardEnc.atleast(lits=X
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


def set_clauses(seq,
                n,
                cnf,
                vpool
                , bound
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

    # potent = get_potential_neighbors_pairs_disjunctions(seq
    #                                                     , n
    #                                                     , vpool
    #                                                     , 1)
    # if potent is not None:
    #     card(potent
    #          , bound
    #          , cnf
    #          )

    # # au moins une valeur par ligne
    # min1value_per_line(size, cnf, vpool)
    # # au moins une valeur par colonne
    # min1value_per_column(size, cnf, vpool)
    return cnf


# print the variables of the solution
def print_solution_variables(seq,
                             n,
                             vpool,
                             sol):
    print("Solution variables:")
    for i in range(n):
        for j in range(n):
            for v in range(n):
                # print("value : ", v)
                # print("value", vpool.id((i, j, v + 1)), "=",
                # sol[vpool.id((i, j, v + 1))])
                if vpool.id((i, j, v)) in sol:
                    print(i, j, seq[v])


def get_matrix(n, vpool, sol):
    matrix = [[0 for x in range(n)] for y in range(n)]
    for i in range(n):
        for j in range(n):
            location_valued = False
            for v in range(n):
                if vpool.id((i, j, v)) in sol:
                    matrix[i][j] = v
                    location_valued = True
            if not location_valued:
                matrix[i][j] = -1
    return matrix


def get_value_matrix(matrix
                     , seq):
    value_matrix = [[0 for i in range(len(matrix))] for j in
                    range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            index = matrix[i][j]
            if index != -1:
                value_matrix[i][j] = seq[index]
            else:
                value_matrix[i][j] = -1
    return value_matrix


def get_score(matrix):
    # score = lambda matrix, n: sum([matrix[i][j] == i + 1 for i in range(n) for j in range(n)])
    score = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            current = matrix[i][j]
            if current == -1:
                # print(i, j, " is -1")
                pass
            else:
                # print(i, j, " is ", current)
                if i + 1 < len(matrix):
                    if current == matrix[i + 1][j]:
                        score += 1
                        # print(i, j, current, "="
                        #       , i + 1, j,
                        #       matrix[i + 1][j])
                        # print("score", score)
                if j + 1 < len(matrix):
                    if current == matrix[i][j + 1]:
                        score += 1
                        # print(i, j, current, "="
                        #       , i, j + 1, matrix[i][j + 1])
                        # print("score", score)
    return score


def print_solution_matrix(matrix
                          , seq
                          , mode="all"):
    print("Solution matrix:")
    for mode in ("index", "value"):
        for i in range(len(matrix)):
            print()
            for j in range(len(matrix)):
                if 0 <= matrix[i][j]:
                    if mode == "index"\
                            or mode == "all":
                        print(matrix[i][j], end=" ")
                    if mode == "value" or mode == "all":
                        print(seq[matrix[i][j]], end=" ")
                else:
                    print("*", end=" ")


def print_value_matrix(matrix):
    print("Solution value matrix:")
    for i in range(len(matrix)):
        print()
        for j in range(len(matrix)):
            print(matrix[i][j], end=" ")


def print_solution(seq,
                   n,
                   vpool,
                   sol
                   , scoring=True):  # affiche la solution
    if scoring:
        score = 0

    # print_solution_variables(seq,
    #                          n,
    #                          vpool,
    #                          sol)
    print()
    print("matrix representation:")
    for i in range(n):
        for j in range(n):
            location_valued = False
            for v in range(n):
                if vpool.id((i, j, v)) in sol:
                    print(seq[v], end=" ")
                    location_valued = True

            if not location_valued: print("* ", end='')
        print()


def solve(seq,
          bound
          ):
    # retourne un plongement de score au moins 'bound'
    # si aucune solution n'existe, retourne None

    # variables ##########################
    vpool = IDPool(
        start_from=1)  # pour le stockage des identifiants entiers des couples (i,j)
    cnf = CNF()  # construction d'un objet formule en forme normale conjonctive (Conjunctive Normal Form)

    n = len(seq)

    # contraintes ##########################
    cnf = set_clauses(seq,
                      n,
                      cnf,
                      vpool
                      , bound)
    print("clauses quantity:", cnf.nv)

    solver = Glucose4(use_timer=True)  # MiniSAT
    # solver = Glucose4(use_timer=True)
    solver.append_formula(cnf.clauses, no_return=False)

    print("Resolution...")
    resultat = solver.solve()
    print("Satisfaisable : " + str(resultat))
    print("Temps de resolution : " + '{0:.2f}s'.format(solver.time()))
    if resultat:

        interpretation = solver.get_model()  # extracting a
        # satisfying assignment for CNF formula given to the solver
        # A model is provided if a previous SAT call returned True.
        # Otherwise, None is reported.
        # Return type list(int) or None

        if interpretation is not None:
            print("Interpretation: ", interpretation)

        # cette interpretation est longue,
        # on va filtrer les valeurs positives
        # (il y a en line_quantity fois moins)
        filtered_interpretation = list(
            filter(lambda x: x >= 0, interpretation))

        matrix = get_matrix(n
                            , vpool
                            , interpretation
                            # , filtered_interpretation
                            )
        value_matrix = get_value_matrix(matrix
                                        , seq)
        affichage_sol = True
        if affichage_sol:
            print("\nVoici une solution: \n")

            # print_value_matrix(value_matrix)
            print_solution_matrix(matrix
                                  , seq
                                  , mode="value")
            # print_solution(seq,
            #                n,
            #                vpool,
            #                filtered_interpretation
            #                # resultat
            #                )
        score = get_score(value_matrix)
        print("score:", score)
        if score >= bound:
            return resultat
    return None


def exist_sol(seq, bound):
    # retourne True si et seulement si il
    # existe un plongement de score au moins 'bound'
    # A COMPLETER
    # clauses = card(X, k)
    if solve(seq,
             bound
             ):
        print("Il existe une solution")
        return True
    return False


# vous pouvez utiliser les methodes de la classe pysat.card pour
# creer des contraintes de cardinalites (au moins k, au plus k,...)

def dichotomy(seq):
    # retourne un plongement de score au moins 'lower_bound' si
    # aucune solution n'existe, retourne None cette fonction utilise
    # la methode de dichotomie pour trouver un plongement de score au
    # moins 'lower_bound' A COMPLETER
    return None


def incremental_search(seq, lower_bound):
    # retourne un plongement de score au moins 'lower_bound'
    # si aucune solution n'existe, retourne None
    # cette fonction utilise une recherche incrémentale
    # pour trouver un plongement de score au moins 'lower_bound'

    bound = lower_bound
    sol = exist_sol(seq, bound)
    while sol:
        sol = exist_sol(seq, bound)
        # tant qu'il existe un plongement de score au moins 'bound'
        bound += 1

    if bound > lower_bound:
        sol = solve(seq, bound - 1)
        return sol
    else:
        print("Pas de solution")
        return None


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


def compute_max_score(seq, method):
    # calcul le meilleur score pour la sequence seq,
    # il doit donc retourne un entier,
    # methode utilisee: dichotomie par defaut,
    #                   incrementale si l'option -i est active
    score_best = 0
    # si l'option -i est active, on utilise la recherche incrémentale
    if method == "incremental":
        score_best = incremental_search(seq, 0)
    else:
        score_best = dichotomy(seq)
    return (score_best)


####################################################################
########### CE CODE NE DOIT PAS ETRE MODIFIE #######################
####################################################################
def test_code():
    examples = [('00', 0),
                ('1', 0),
                ('01000', 0),
                ('00110000', 1), ('11', 1), ('111', 2), ('1111', 4),
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
                ('10110011010010001110', 11)]
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
        try:
            if func_timeout.func_timeout(TIMEOUT, exist_sol,
                                         [seq, bound]):
                sat_tests_success += 1
                print(" ---> succes")
            else:
                print(" ---> echec")
        except func_timeout.FunctionTimedOut:
            timeouts_sat_tests += 1
            print(" ---> timeout")
        except Exception as e:
            exceptions_sat_tests += 1
            print(" ---> exception levee")

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
            else:
                print(" ---> echec")
        except func_timeout.FunctionTimedOut:
            timeouts_unsat_tests += 1
            print(" ---> timeout")
        except Exception as e:
            exceptions_unsat_tests += 1
            print(" ---> exception levee")

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

exist_sol("01", 0)
exist_sol("11", 0)

if test:
    print("Let's test your code")
    test_code()
elif options.bound != None:
    # cas ou la borne est fournie en entree:
    # on test si la sequence (qui doit etre donnee en entree) a un score superieur ou egal a la borne donnee
    # si oui, on affiche "SAT".
    # Si l'option d'affichage est active,
    #   alors il faut egalement afficher une solution
    print("DEBUT DU TEST DE SATISFIABILITE")
    # A COMPLETER

    if exist_sol(options.sequence, options.bound):
        print("SAT")
        if options.display:
            print_solution()

    print("FIN DU TEST DE SATISFIABILITE")

elif not (incremental):
    # on affiche le score maximal qu'on calcule par dichotomie
    # si l'option d'affichage est active
    #   on affiche egalement un plongement de score maximal
    print("DEBUT DU CALCUL DU MEILLEUR SCORE PAR DICHOTOMIE")

    test_code()

    if len(sys.argv) > 1:
        print(
            "Calcul du meilleur score pour la sequence " + options.sequence)
        compute_max_score(options.sequence
                          , method="dichotomy"
                          )
    print("FIN DU CALCUL DU MEILLEUR SCORE")

elif not test:
    # Pareil que dans le cas precedent mais avec la methode incrementale
    # A COMPLETER
    print("DEBUT DU CALCUL DU MEILLEUR SCORE PAR METHODE INCREMENTALE")
    # A COMPLETER
    print("FIN DU CALCUL DU MEILLEUR SCORE")

import sys
import numpy
from pysat.solvers import Minisat22
# from pysat.solvers import Glucose4
# from pysat.formula import CNF
# from pysat.formula import IDPool
from pysat.card import *
from optparse import OptionParser
import func_timeout

from auto_indent import *

sys.stdout = AutoIndent(sys.stdout)


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

def set_neighbors(matrix_size,
                  vpool,
                  sequence_index1,
                  sequence_index2
                  , to_append
                  , neighborhood_symbol=None
                  ):
    # print("set_neighbors", a, b)
    # deux points (i, j), (k, l) ∈ N² sont voisins si
    # (|i − k|, |j − l|) ∈ {(0, 1), (1, 0)}.
    for i in range(matrix_size):
        for j in range(matrix_size):
            d = [-vpool.id((i, j, sequence_index1))]
            for k in range(matrix_size):
                for m in range(matrix_size):
                    print("i = ", i
                          , "\nj = ", j
                          , "\nk = ", k
                          , "\nl = ", m)
                    if are_neighbors(i, j, k, m):
                        # print("              i, j, k, l", i, j, k, l)
                        d.append(vpool.id((k, m, sequence_index2)))
                        print("i", i, "j", j, "sequence_index1",
                              sequence_index1)
                        print("k", k, "m", m, "sequence_index2", sequence_index2)
                        if neighborhood_symbol is not None:
                            print("neighborhood_symbol =",
                                  neighborhood_symbol)

                            # neighborhood_symbol <-> neighborhood

                            # neighborhood_symbol -> neighborhood &
                            # neighborhood_symbol <- neighborhood

                            # -neighborhood_symbol | neighborhood &
                            # neighborhood_symbol | neighborhood

                            print()

                            to_append.append([-neighborhood_symbol,
                                              vpool.id((i
                                                        , j
                                                        , sequence_index1))])
                            print("[-neighborhood_symbol vpool.id((i "
                                  "j sequence_index1))]")
                            print("not neighborhood_symbol", "-",
                                  neighborhood_symbol)
                            print(vpool.id((i, j, sequence_index1)))
                            print()

                            to_append.append([neighborhood_symbol,
                                              -vpool.id((i
                                                         , j
                                                         , sequence_index1))])
                            print()
                            print("neighborhood_symbol",
                                  neighborhood_symbol)
                            print("not vpool.id((k m "
                                  "sequence_index2))")
                            print(k, m, sequence_index2)

                            to_append.append(
                                [-neighborhood_symbol,
                                 vpool.id((k
                                           , m
                                           , sequence_index2
                                           ))])
                            print("not neighborhood_symbol", "-",
                                  neighborhood_symbol)
                            print(vpool.id((k, m, sequence_index2)))
                            print()
                            to_append.append(
                                [neighborhood_symbol,
                                 -vpool.id((k
                                            , m
                                            , sequence_index2
                                            ))])
                            print(neighborhood_symbol)
                            print("not vpool.id((k m "
                                  "sequence_index2))", "-", vpool.id((k
                                            , m
                                            , sequence_index2
                                            )))
                            neighborhood_symbol += 1
            # print("d = ", d)
            to_append.append(d)
    if neighborhood_symbol is not None:
        return neighborhood_symbol
    else:
        return to_append


neighbors_cnf = CNF()
neighbors_vpool = IDPool()

set_neighbors(2
              , neighbors_vpool
              , 0
              , 1
              , neighbors_cnf
              , 1
              )
print(neighbors_cnf.clauses)

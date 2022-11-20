from pysat.solvers import Glucose4
from pysat.formula import CNF
from pysat.formula import IDPool
from pysat.card import *
from pysat.card import ITotalizer


# classmethod atleast(lits, bound=1, top_id=None, vpool=None, encoding=1)

# permet, étant donné une liste de litéraux 'lits' et une borne 'bound',
# de créer une formule en CNF (autrement dit, un ensemble de clauses)
# qui sera satisfaisable uniquement par des valuations telles que
# au moins 'bound' litéraux de lits sont vrais.
# L'argument 'vpool' permet de préciser quel
# "créateur" de variables vous utilisez
# (c'est utile pour que la méthode ne crée pas des variables qui serait
# déjà utilisés dans d'autres parties de votre programme).
# L'argument top_id sequence_length'est a priori pas utile, et
# l'argument 'encoding' permet de changer l'algorithme utiliser
# pour créer cet ensemble de clauses.
# Prenons un exemple. Supposons qu'on se donne un
# entier sequence_length et qu'on veuille
# remplir un tableau de sequence_length cases avec des pions de telle sorte que
# 1) il sequence_length'y jamais deux pions voisins
# 2) il y a au moins 5 pions dans le tableau
# On peut naturellement prendre l'ensemble des variables
# X1,..., Xn avec la sémantique
# "Xi est vraie ssi il y a un pion dans la case i".
# Alors, la première contrainte 1) se traduit par la formule:

# F1 = (non x1 ou non x2) et (non x2 ou non x3) et (non x3 ou non x4)
# ...
def no_neighbor(n):
    return 0


# Pour la deuxième contrainte, on utilise une contrainte de
# cardinalité: F2 = atleast([X1,...,Xn], 5)
# Puis on demande à tester
# la satisfaisabilité de la formule "F1 and F2".

def sattest(n):
    myvpool = IDPool(start_from=1)
    cnf = CNF()
    lits = [0]
    for i in range(n):
        lits.append(myvpool.id(i))
        if i < n - 1:
            # cnf.add_clause([myvpool.id(i), myvpool.id(i + 1)])
            cnf.append([-myvpool.id(i), -myvpool.id(i + 1)])
    cnf.extend(CardEnc.atleast(lits
                               # , 2
                               # , 3
                               , 5
                               , vpool=myvpool
                               # , encoding=EncType.seqcounter
                               , encoding=1
                               ))


# sattest(2) ValueError: non-zero integer expected
# sattest(3)
# sattest(5)
sattest(15)

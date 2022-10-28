from pysat.card import *
from pysat.card import ITotalizer

# bound = 1
# Note that the implementation of the
# pairwise,
# bitwise,
# and ladder
# encodings can only deal with AtMost1 constraints
cnf = CardEnc.atmost(lits=[1, 2, 3],
                     encoding=EncType.pairwise)

print(cnf.clauses)
# [[-1, -2], [-1, -3], [-2, -3]]


cnf = CardEnc.equals(lits=[1, 2, 3],
                     encoding=EncType.pairwise)
print(cnf.clauses)
# [[1, 2, 3],
# [-1, -2], [-1, -3], [-2, -3]]


cnf = CardEnc.atmost(lits=[1, 2, 3],
                     bound=2,
                     encoding=EncType.pairwise)
print(cnf.clauses)
# [[-1, -2, -3]]
#

cnf = CardEnc.atmost(lits=[1, 2, 3],
                     bound=6,
                     encoding=EncType.pairwise)
print(cnf.clauses)
# []

cnf = CardEnc.atmost(lits=[1, 2, 3],
                     bound=6,
                     encoding=EncType.pairwise)
print("lits_quantity: " + str(cnf.nv) )
print(cnf.clauses)

cnf = CardEnc.atleast(lits=[1, 2, 3],
                     bound=2,
                     encoding=EncType.pairwise)
print("lits_quantity: " + str(cnf.nv) )
print(cnf.clauses)

cnf = CardEnc.atleast(lits=[1, 2, 3],
                     bound=3,
                     encoding=EncType.pairwise)
print("lits_quantity: " + str(cnf.nv) )
print(cnf.clauses)

cnf = CardEnc.atleast(lits=[1, 2, 3],
                     # bound=4,# raise ValueError('Wrong bound: {0}'.format(bound))
# ValueError: Wrong bound: 4
                     encoding=EncType.pairwise)
print("lits_quantity: " + str(cnf.nv) )
print(cnf.clauses)


with ITotalizer(lits=[1, 2, 3], ubound=1) as t:
    print(t.cnf.clauses)
# [[-2, 4], [-1, 4], [-1, -2, 5], [-4, 6], [-5, 7], [-3, 6], [-3, -4, 7]]
    print(t.rhs)
# [6, 7]

with ITotalizer(lits=[1, 2, 3, 66], ubound=1) as t:
    print(t.cnf.clauses)
# [[-2, 4], [-1, 4], [-1, -2, 5], [-4, 6], [-5, 7], [-3, 6], [-3, -4, 7]]
    print(t.rhs)
# [6, 7]

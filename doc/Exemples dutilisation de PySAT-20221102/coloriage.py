#!/usr/bin/env python3.6
# cours informatique fondamentale 2021-2022
# chapitre SAT
# probleme des huit reines


from pysat.solvers import Minisat22
from pysat.solvers import Glucose4
from pysat.formula import CNF
from pysat.formula import IDPool

vpool = IDPool(start_from=1) # pour le stockage des identifiants entiers des couples (i,j)

cnf = CNF()  # construction d'un objet formule en forme normale conjonctive (Conjunctive Normal Form)


# construction de la formule

regions = [1,2,3,4,5,6]
topologie = [(1,2),(1,3),(2,3),(2,4),(3,4),(3,5),(3,6),(4,5),(5,6)]
couleurs = [1,2,3]


print("Construction des clauses\n")

# Jamais deux regions voisines avec la meme couleur

for r1 in regions:
    for (x,r2) in topologie:
        for c in couleurs:
            if (x==r1):
                cnf.append([-vpool.id((r1,c)),-vpool.id((r2,c))])


# au moins une couleur par region
for r in regions:
    d = []
    for c in couleurs:
        d.append(vpool.id((r,c)))
    cnf.append(d)

print("Clauses construites:\n")
print(cnf.clauses) # pour afficher les clauses
print("\n")

# phase de resolution

s = Minisat22() # pour utiliser le solveur MiniSAT
# s = Glucose4(use_timer=True) # pour utiliser le solveur Glucose
s.append_formula(cnf.clauses, no_return=False)

print("Resolution...")
resultat = s.solve()
print("satisfaisable : " + str(resultat))

# affichage solution

if resultat:

    print("\nVoici une solution: \n")

    model = s.get_model()

    for r in regions:
        for c in couleurs:
            if vpool.id((r,c)) in model:
                print("Region " + str(r) + " coloriee avec couleur " + str(c) + "\n")
        

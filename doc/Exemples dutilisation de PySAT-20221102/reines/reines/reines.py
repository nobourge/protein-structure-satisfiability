#!/usr/bin/env python3.6
# cours informatique fondamentale 2021-2022
# chapitre SAT
# probleme des huit reines


from pysat.solvers import Minisat22
from pysat.solvers import Glucose4
from pysat.formula import CNF
from pysat.formula import IDPool


nb = 8 # nombre de reines

affichage_sol = True # affichage d'une solution

vpool = IDPool(start_from=1) # pour le stockage des identifiants entiers des couples (i,j)

cnf = CNF()  # construction d'un objet formule en forme normale conjonctive (Conjunctive Normal Form)


# construction de la formule

print("Construction des clauses\n")

print("Au moins un reine par ligne")


txt = "clauses quantity:"
print(f'{txt, cnf.nv}')
print()

# au moins un reine par ligne
for i in range(nb):
    d = []
    for j in range(nb):
        d.append(vpool.id((i,j)))
    cnf.append(d)

txt = "clauses quantity:"
print(f'{txt, cnf.nv}')
print()

# pas d'attaque en ligne
print("Pas d'attaque en ligne")

for i in range(nb):
    for j in range(nb):
        for k in range(nb):
            if k > j:
                cnf.append([-vpool.id((i,j)),-vpool.id((i,k))])
print("done")

txt = "clauses quantity:"
print(f'{txt, cnf.nv}')
print()

# pas d'attaque en colonne
print("Pas d'attaque en colonne")

for i in range(nb):
    for j in range(nb):
        for k in range(nb):
            if k > j:
                cnf.append([-vpool.id((j,i)),-vpool.id((k,i))])
print("done")
           
txt = "clauses quantity:"
print(f'{txt, cnf.nv}')
print()

# pas d'attaque en diagonale 

print("Pas d'attaque en diagnale")

# version lente
# for i in range(nb):
    # for j in range(nb):
        # for k in range(nb):
            # for l in range(nb):
                # if abs(i-k)==abs(j-l) and i!=k:
                    # cnf.append([-vpool.id((i,j)),-vpool.id((k,l))])
# print("done")

txt = "clauses quantity:"
print(f'{txt, cnf.nv}')
print()

# version optimisee
for i in range(nb):
    for j in range(nb):
        for k in range(nb-1):
                if i+k+1<nb and j+k+1<nb:
                    cnf.append([-vpool.id((i,j)),-vpool.id((i+k+1,j+k+1))])
                    

for i in range(nb):
    for j in range(nb):
        for k in range(nb-1):
                if i+k+1<nb and j-k-1>=0:
                    cnf.append([-vpool.id((i,j)),-vpool.id((i+k+1,j-k-1))])
                    
print("done")

# print(cnf.clauses) # pour afficher les clauses


# phase de resolution

s = Minisat22(use_timer=True) # pour utiliser le solveur MiniSAT
# s = Glucose4(use_timer=True) # pour utiliser le solveur Glucose
s.append_formula(cnf.clauses, no_return=False)

print("Resolution...")
resultat = s.solve()
print("satisfaisable : " + str(resultat))

print("Temps de resolution : " + '{0:.2f}s'.format(s.time()))

# affichage solution

if affichage_sol and resultat:

    print("\nVoici une solution: \n")
    print(s.get_model())
    
    for i in range(nb):
        for j in range(nb):
            if vpool.id((i,j)) in s.get_model():
                print("R ",end='')
            else:
                print("* ",end='')
        print("")

# obtention d'une autre solution

d = []
for v in s.get_model():
    d.append(-v)
cnf.append(d)
s.append_formula(cnf.clauses, no_return=False)

resultat = s.solve()
print("satisfaisable : " + str(resultat))

print("Temps de resolution : " + '{0:.2f}s'.format(s.time()))

# affichage solution

if affichage_sol and resultat:

    print("\nVoici une solution: \n")
    print("valuation trouvee : " + str(s.get_model()))
    
    for i in range(nb):
        for j in range(nb):
            if vpool.id((i,j)) in s.get_model():
                print("R ",end='')
            else:
                print("* ",end='')
        print("")



# obtenir une deuxieme solution
if resultat:
    print("calcul d'une autre solution si possible")
    d = []
    for v in s.get_model():
        d.append(-v)
    cnf.append(d)
    s.append_formula(cnf.clauses, no_return=False)

    resultat = s.solve()
    print("satisfaisable : " + str(resultat))

    print("Temps de resolution : " + '{0:.2f}s'.format(s.time()))

    if affichage_sol and resultat:

        for i in range(nb):
            for j in range(nb):
                if vpool.id((i,j)) in s.get_model():
                    print("R ",end='')
                else:
                    print("* ",end='')
            print("")

# cours informatique fondamentale 2021-2022
# chapitre SAT
# probleme des pavages de Wang

import sys
from pysat.solvers import Minisat22
from pysat.solvers import Glucose4
from pysat.formula import CNF
from pysat.formula import IDPool
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter
import numpy

# PARAMETRES

dim = 50 # pavage d'un carre dim x dim
tile_width = 40 # largeur des tiles
tile_height = 40 # hauteur des tiles
affichage_sol_txt = False # affichage de la premier solution en mode textuel
affichage_sol_graphics = True # affichage solution(s) en image
nb_solutions = 10 # nombre de solutions differentes a afficher

########################

tile_dimension = (tile_width,tile_height)

# object Tiles

class Tile:
    def __init__(self, north, south, west, east, path, name):
        self.north = north
        self.south = south
        self.west = west
        self.east = east
        self.img = Image.open(path).resize(tile_dimension,resample=Image.BICUBIC) # ouver l'image et la redimensionne
        self.name = name
        
    def __str__(self):
        return f"North : {self.north}, South : {self.south}, West : {self.west}, East : {self.east}"

    def shortstr(self):
        return f"{self.north}{self.south}{self.west}{self.east}"

    


vpool = IDPool(start_from=1) # pour le stockage des identifiants entiers des couples (i,j)
cnf = CNF()  # construction d'un objet formule en forme normale conjonctive (Conjunctive Normal Form)

# CREATION DE L'ENSEMBLE DES TILES

## COURBES
# tile1 = Tile(1, 2, 3, 4, 'images/tube/1.png', '1')
# tile2 = Tile(1, 2, 3, 4, 'images/tube/2.png', '2')
# tile3 = Tile(2, 1, 4, 3, 'images/tube/3.png', '3')
# tile4 = Tile(2, 1, 4, 3, 'images/tube/4.png', '4')

# tile_list = [tile1, tile2, tile3, tile4]

## PAVAGE APERIODIQUE
R = 1
V = 2
B = 3
W = 4
tile1 = Tile(R, R, V, R, 'images/aperiodique/1.png', '1')
tile2 = Tile(B, B, V, R, 'images/aperiodique/2.png', '2')
tile3 = Tile(R, V, V, V, 'images/aperiodique/3.png', '3')
tile4 = Tile(W, R, B, B, 'images/aperiodique/4.png', '4')
tile5 = Tile(B, W, B, B, 'images/aperiodique/5.png', '5')
tile6 = Tile(W, R, W, W, 'images/aperiodique/6.png', '6')
tile7 = Tile(R, B, W, V, 'images/aperiodique/7.png', '7')
tile8 = Tile(B, B, R, W, 'images/aperiodique/8.png', '8')
tile9 = Tile(B, W, R, R, 'images/aperiodique/9.png', '9')
tile10 = Tile(V, B, R, V, 'images/aperiodique/10.png', '10')
tile11 = Tile(R, R, V, W, 'images/aperiodique/11.png', '11')

tile_list = [tile1, tile2, tile3, tile4, tile5, tile6, tile7, tile8, tile9, tile10, tile11]


# construction de la formule

print("Construction des clauses\n")

print("Au moins un tile par case\n")

# i varie sur les colonnes
# j varie sur les lignes
# (0,0) correspond au coin nord-ouest
# (dim-1,dim-1) correspond au coin sud-est

for i in range(dim):
    for j in range(dim):
        d = []
        for t in tile_list:
            d.append(vpool.id((i,j,t)))
        cnf.append(d)

# tout tile doit apparaitre au moins une fois
        
# for t in tile_list:         
    # d = []
    # for i in range(dim):
        # for j in range(dim):
            # d.append(vpool.id((i,j,t)))
    # cnf.append(d)



print("Au plus un tile par case\n")

for i in range(dim):
    for j in range(dim):
        for t1 in tile_list:
            for t2 in tile_list:
                if not (t1 is t2):
                    cnf.append([-vpool.id((i,j,t1)),-vpool.id((i,j,t2))])
                    
print("Les faces coincident verticalement\n")
                    
for i in range(dim-1):
    for j in range(dim):
        for t1 in tile_list:
            for t2 in tile_list:
                if (t1.east != t2.west):
                    cnf.append([-vpool.id((i,j,t1)),-vpool.id((i+1,j,t2))])


                    
print("Les faces coincident horizontalement\n")
                    
for i in range(dim):
    for j in range(dim-1):
        for t1 in tile_list:
            for t2 in tile_list:
                if (t1.south != t2.north):
                    cnf.append([-vpool.id((i,j,t1)),-vpool.id((i,j+1,t2))])

                    
# print(cnf.clauses) # pour afficher les clauses


# phase de resolution

# s = Minisat22(use_timer=True) # pour utiliser le solveur MiniSAT
s = Glucose4(use_timer=True) # pour utiliser le solveur Glucose
s.append_formula(cnf.clauses, no_return=False)


print("Resolution...")
sat = s.solve()
print("satisfaisable : " + str(sat))


print("Temps de resolution : " + '{0:.2f}s'.format(s.time()))


# affichage solution


if affichage_sol_txt or affichage_sol_graphics:
    model = s.get_model()

if affichage_sol_txt and sat:

    print("\nVoici une solution: \n")

    
    for i in range(dim):
        for j in range(dim):
            for t in tile_list:
                if vpool.id((i,j,t)) in model:
                    print(t.shortstr() + " ",end='')
            if j == dim-1:
                print("")

# affichage graphique d'une solution
if affichage_sol_graphics and sat:
    new_width = dim*tile_width
    new_height = dim*tile_height
    plt.ion() # activer le mode interactif d'affichage (pour que le meme graphique soit mis a jour a chaque nouvelle solution trouvee)

    dst = Image.new('RGB', (new_width, new_height))

    for i in range(dim):
        for j in range(dim):
            for t in tile_list:
                if vpool.id((i,j,t)) in model:
                    dst.paste(t.img, (tile_width*i, tile_height*j))
    plt.imshow(dst)
    plt.show()
    plt.pause(0.0001)
    input("")
    # dst.show()

# obtention de nouvelles solutions et affichage
    iterations = nb_solutions-1

    for it in range(iterations):
        # contraintes pour avoir une solution differente
        d = [] 
        for l in model: 
            d.append(-l) # negation de l
        new_constraints = CNF()
        new_constraints.append(d) # ajout d'une clause qui demande a ce que la nouvelles solution si elle existe differe a au moins un endroit
        s.append_formula(new_constraints,no_return=False) # mise a jour du solveur
        sat = s.solve() # resolution
        if sat:
            model = s.get_model()
            for i in range(dim):
                for j in range(dim):
                    for t in tile_list:
                        if vpool.id((i,j,t)) in model:
                            dst.paste(t.img, (tile_width*i, tile_height*j))
        
            plt.imshow(dst) 
            plt.show() # affichage
            plt.pause(0.0001)
            input("Appuyer sur Enter pour avoir une autre solution")
        else:
            print("UNSAT")


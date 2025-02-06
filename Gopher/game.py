# - IMPORT - 
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

# - JEU

# -- Type --
GridSize = int
Grid = tuple[tuple[int, ...], ...]
State = Grid

Pos = tuple[int, int]
Pion = Pos
Action = Pos
Player = int
Score = float
Strategy = Callable[[Grid, Player], Action]

# -- Constantes --
VIDE = 0
ROUGE = 1
BLEU = -1
 

class Gopher:
    """
    Plateau de forme hexagonale, composé d'hexagones
    pour l'instant on utilise une structure de donnée sous forme de tableau
    joueur rouge = 1, bleu = 2
    """

    def __init__(self, taille_grid:GridSize):
        self.taille = taille_grid

    # Changement de type de grille
    def gridTupleToGridList(self, grid: Grid) -> list[list[int]]:
        return list(list(subliste) for subliste in grid) 

    def gridListToGridTuple(self, grid: list[list[int]]) -> Grid:
        return tuple(tuple(subliste) for subliste in grid)

    # construction de la grille
    def create_grid(self) -> Grid:
        # prend une taille de grille en entré qui coresspond au nombre d'exagone par côté 
        # Retourne un Type grid ou chaque ligne correspond à une ligne verticale d'exagone

        # On exclue évidement le cas ou la grille fait une seule case
        if self.taille == 1:
            raise ValueError("La taille de la grille doit être minimum de 2. On rappel que la taille correspond au nombre d'exagone par côté")
        
        grille = []
        for i in range(self.taille-1, -self.taille, -1):
            ligne = []
            for j in range(-self.taille+1, self.taille):
                if  i-j-self.taille+1 > 0 or j - i - self.taille+1 > 0  :
                    ligne.append(None)
                else:
                    ligne.append((0))
            grille.append(ligne)
        return self.gridListToGridTuple(grille)

    # affichage de la grille
    def pprint(self, grid) -> None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')

        taille = int((len(grid) + 1) / 2)
        flat_list = []
        
        for ligne in grid:
            for val in ligne:
                if val != None:
                    flat_list.append(val)

        x_coords = []
        y_coords = []
        taille -= 1
        for q in range(-taille, taille + 1):
            r1 = max(-taille, -q - taille)
            r2 = min(taille, -q + taille)
            for r in range(r1, r2 + 1):
                x_coords.append(q * 3 / 2)
                y_coords.append(np.sqrt(3) * (r + q / 2))
            
        x_rot_coords = []
        y_rot_coords = []    
        # Angle d'inclinaison d'un douzième de tour en radians
        angle_inclinaison = np.pi / 6
        for x, y, value in list(zip(x_coords, y_coords, flat_list)):
            # Rotation des coordonnées x et y autour de l'origine
            x_rot_coords.append(x * np.cos(angle_inclinaison) - y * np.sin(angle_inclinaison))
            y_rot_coords.append(x * np.sin(angle_inclinaison) + y * np.cos(angle_inclinaison))

        # Combinaison des coordonnées en tuples (x, y)
        zipped_coords = zip(x_rot_coords, y_rot_coords)

        # Trier les coordonnées selon x croissant et y décroissant
        sorted_coords = sorted(zipped_coords, key=lambda zipped_coords: (-zipped_coords[1], zipped_coords[0]))
        x_sorted_coords, y_sorted_coords = zip(*sorted_coords)

        for x, y, value in list(zip(x_sorted_coords, y_sorted_coords, flat_list)):
            color_map = {0: 'white', 1: 'red', -1: 'blue'}
                            
            circle = plt.Circle((x, y), 0.8, color= color_map.get(value), ec='black', lw=1)
            ax.add_patch(circle)
                
        ax.autoscale_view()
        ax.axis('off')
        plt.show()
        plt.close()

    # Verification que la grid contient une valeur à un indice positif donné
    def is_in_range(self, grid: Grid, pos: Pos) -> bool:
        ligne = pos[0]
        colonne = pos[1]
        if ligne < 0 or colonne < 0:
            return False
        elif (ligne < 0) or (ligne > len(grid)-1):
            return False
        elif (colonne < 0) or (colonne > len(grid[ligne])-1):
            return False
        elif grid[ligne][colonne] == None:
            return False
        else:
            return True

    # Verifie qu'il y a une connection ennemy au joueur pour une position donné
    def is_ennemy_connection(self, grid: Grid, pos: Pos, joueur: Player) -> bool:
        # on parcours toutes les cases adjacentes:
        for i in range(-1, 2):
            for j in range (-1, 2):
                if self.is_in_range(grid, (pos[0]+i, pos[1]+j)):
                    if (i, j) != (0, 0) and not (j == i) and grid[pos[0]+i][pos[1]+j] == -joueur:
                        return True
        return False

    # Verifie qu'il y a une connection allié au joueur pour une position donné
    def is_friendly_connection(self, grid: Grid, pos: Pos, joueur: Player) -> bool:
        # on parcours toutes les cases adjacentes:
        for i in range(-1, 2):
            for j in range(-1, 2):
                if self.is_in_range(grid,(pos[0]+i, pos[1]+j)):
                    if (i, j) != (0, 0) and not (j == i)  and grid[pos[0]+i][pos[1]+j] == joueur:
                        return True
        return False

    # Renvoie si la grille est vide ou non
    def is_grid_empty(self, grid: Grid) -> bool:
        return not any(1 in ligne or -1 in ligne for ligne in grid)

    # Obtention des coups possible
    def legals(self, grid: Grid, player: Player)-> tuple[Action, ...]:
        """
        On parcours l'ensemble des cases du plateau et on verifie qu'il y a une connection
        ennemie et aucune connection allié 
        """
        if self.is_grid_empty(grid):
            return self.list_case()
        list_action: list[Pos] = []
        for i, ligne in enumerate(grid):
            for j,val in enumerate(ligne):
                if val != None:
                    # print(f"Val : {val}, pos : {(i, j)}, Ennemie : {self.is_ennemy_connection(grid, (i,j), player)}, Allié : {self.is_friendly_connection(grid, (i,j), player)}")
                    if val == 0 and self.is_ennemy_connection(grid, (i,j), player) and self.is_friendly_connection(grid, (i,j), player) == False:
                        list_action.append((i,j))
        return tuple(list_action)
    
    # Donne la liste des cases de la grille
    def list_case(self) -> list[Pos]:
        list_case: list[Pos] = []
        grid = self.create_grid()
        for i, ligne in enumerate(grid):
            for j, val in enumerate(ligne):
                if val != None:
                    list_case.append((i,j))
        return list_case
    
    # Permet d'obtenir la liste des coups valides sous forme de liste de bouléen
    # On peut retrouver les actions en utilisant cette liste comme masque par 
    # rapport à la liste des cases
    def legals_bool(self, grid: State, joueur: Player) -> list[bool]:
        legals = self.legals(grid, joueur)
        list_bool: list[bool] = []
        list_case = self.list_case()
        for case in list_case:
            if case in legals:
                list_bool.append(1)
            else:
                list_bool.append(0)
        return list_bool

    def get_indice_action_in_legal_bool(self, action)-> int:
        list_action = self.list_case()
        indice = list_action.index(action)
        return indice 

    # renvoie le joueur ennemie
    def get_ennemie(self, joueur: Player) -> Player:
        return -joueur 

    # Jouer un coup
    def play(self, grid: State, coup: Action, joueur: Player) -> State:
        tmp_grid = self.gridTupleToGridList(grid)
        tmp_grid[coup[0]][coup[1]] = joueur
        return self.gridListToGridTuple(tmp_grid)

    # Determiner si la partie est fini
    def final(self, grid: State, au_tour_de: Player) -> bool:
        if self.legals(grid, au_tour_de) == ():
            return True
        else :
            return False

    # Donner le score, si on rajoute que l'on souhaite le score de l'ennemie, on l'inverse
    def score(self, grid: State, au_tour_de: Player, get_ennemie_score: bool = False) -> Score:
        score: Score
        if self.final(grid, au_tour_de):
            score = -au_tour_de
        else:
            score = 0
        if get_ennemie_score:
            return -score
        else:
            return score

    # Prend un etat objectif et renvoie la perspective du jeu en fonction du joueur (10% rapide en natif qu'avec numpy)
    def changer_perspective(self, grid: State, joueur: Player) -> State:
        if joueur == -1:
            new_grid = []
            for ligne in grid:
                new_ligne = []
                for val in ligne:
                    if val == None:
                        new_ligne.append(None)
                    else:
                        new_ligne.append(val * joueur)
                new_grid.append(new_ligne)
            return new_grid
        return grid
    
    # Prend un état et l'encode pour le donner à un reaseau de neuronne convolutionnel
    def encode_state_for_convnn(self, state: State) -> list[list[list]]:
        # on fait trois couches de grille pour le reseau convolutionnel
        # 1- 1 si la case est possédé par le joueur bleu sinon 0   
        # 2- 1 si case libre sinon 0
        # 3- 1 si la case est possédé par le joueur rouge sinon 0   
        
        encoded_grid = []
        for i in range(-1, 2):
            i_grid = []
            for ligne in state:
                i_ligne = []
                for val in ligne:
                    if val == i :
                        i_ligne.append(1)
                    else:
                        i_ligne.append(0)
                i_grid.append(i_ligne)
            encoded_grid.append(i_grid)

        encoded_grid = np.stack(
            (encoded_grid[0],encoded_grid[1],encoded_grid[2])
        ).astype(np.float32)
        
        return encoded_grid

    # prend un etat et le flat pour pouvoir le donner à un réseau de neurones basique (fcnn) 
    def encode_flat_state_for_fcnn(self, state: State) -> list:
        flat_list = []
        for ligne in state:
            for val in ligne:
                if val != None:
                    flat_list.append(val)
        return np.array(flat_list).astype(np.float32)

    # donne le nombre d'action total sur le plateau
    def get_nb_actions(self) -> int:
        t = self.taille
        # l représente la taille de la plus grande ligne du plateau
        l = (2*t)-1
        return ((l*l) - (t*(t-1)))
    
    # Même que le nombre d'action mais permet d'être plus eplicite et modulaire dans l'utilisation 
    def get_nb_case(self) -> int:
        t = self.taille
        # l représente la taille de la plus grande ligne du plateau
        l = (2*t)-1
        return ((l*l) - (t*(t-1)))

    def get_nb_case_including_none(self) -> int:
        t = self.taille
        #  correspond à la plus grande ligne du plateau
        l = (2*t -1)
        return l*l

    def get_action_par_indice(self, indice: int) -> Action:
        list_case = self.list_case()
        action = list_case[indice]
        return action

    # Calcul des pions d'une couleurs
    def nb_pion_on_grid(self, state:State, joueur:Player) -> int:
        nb = 0
        for ligne in state:
            for num in ligne:
                if num == joueur:
                    nb += 1
        return nb


    def jouer_partie(self,
        stategy_rouge: Strategy,
        strategy_bleu: Strategy,
        debug: bool = False,
        affichage_graphique:bool = False
        ) -> Score:
        
        grid: Grid = self.create_grid()
        tour = 1
        while self.final(grid, tour) != True:
            if debug:
                couleur = "rouge" if tour == 1 else "bleu"
                print(f"C'est un tour du joueur {couleur}, l'état de la grille est :")
                print(grid)
            if affichage_graphique:
                self.pprint(grid)

            strategy = stategy_rouge if tour == 1 else strategy_bleu
            coup = strategy(grid, tour, self)
            legals = self.legals(grid, tour)
            if coup not in legals:
                raise ValueError(f"Le joueur {tour}, a tenté un coup illegal !!!")
            grid = self.play(grid, coup, tour)
            tour = 1 if tour == -1 else -1
        
        if debug:
            score_partie = self.score(grid, tour)
            gagnant = "rouge" if score_partie == 1 else "bleu"
            print(f"La partie est finit, le gagnant est le joueur {gagnant} voici le dernier plateau : ")
            self.pprint(grid)
        
        return self.score(grid, tour)


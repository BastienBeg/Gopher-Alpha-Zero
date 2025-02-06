# - Import -

from game import * 
import random
import math
import numpy as np
import torch

# Paramètre de torch
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Type hyperparamètres

"""
hyperparametre_dict= {
    "nb_simulation" : 90,
    "UBC_coef_C" : 1.41,
    "nb_cycle" : 4,
    "nb_partie_solo" : 50,
    "epochs" : 3,
    "batch_size" : 16,
    "temperature": 1.25,
    "dirichlet_epsilon": 0.25,
    "dirichlet_alpha": 0.3,
    "learning_rate" : 0.01,
    "weight_decay" :  0.0001,
    "nom_model" : "model_resnet-2",
    "type_model" : "resnet",
    "num_res_blocks" : 4,
    "num_hidden_layers" : 32
    }
"""

# - Classes -

class alpha_Node:
    def __init__(self, game: Gopher, 
                 grid: State, 
                 hyperparametre: dict, 
                 parent = None, 
                 derniere_action: Action = None, 
                 prior: float = 0, 
                 nb_visite: int = 0) -> None:
        
        self.game = game
        self.grid = grid
        self.parent = parent
        self.derniere_action = derniere_action
        self.prior = prior
        self.hyperparametre = hyperparametre

        self.nb_visite = nb_visite
        self.nb_victoire = 0
        self.children = []

    def is_expand(self) -> bool:
        if len(self.children) > 0:
            return True
        else:
            return False
    
    def score_ucb(self, child) -> float:
        # usuellement le taux de victoire correspond à victoire/visite
        # néanmoins selon l'implémentation il peut être négatif donc on le
        # normalise, d'ou le +1 et /2. Puis on prend l'inverse via le 1- 
        # car la perspective est toujour celle du joueur 1 dans l'arbre
        param_c = self.hyperparametre['UBC_coef_C']

        if child.nb_visite == 0:
            taux_victoire = 0
        else: 
            taux_victoire = 1 - (((child.nb_victoire/child.nb_visite) + 1)/2)

        coef_exploration =child.prior * param_c * math.sqrt( ((self.nb_visite)) / (child.nb_visite + 1) )
        return taux_victoire + coef_exploration
    

    def select_node(self):
        best_child = None
        best_ucb = -100.0
        for child in self.children:
            child_ucb = self.score_ucb(child)
            if child_ucb > best_ucb:
                best_ucb = child_ucb
                best_child = child
        if best_ucb == -100.0:
            raise ValueError("Problème d'ucb négatif")
        return best_child

    def etendre(self, policy: list):
        for action_indice, prob in enumerate(policy):
            if prob > 0:
                action = self.game.get_action_par_indice(action_indice)

                child_grid = list(self.grid)
                child_grid = child_grid.copy()
                child_grid = self.game.play(child_grid, action, 1)
                child_grid = self.game.changer_perspective(child_grid, -1)
                child = alpha_Node(self.game, child_grid, self.hyperparametre,self, action, prob)
                self.children.append(child)

        return child

    def backpropagation(self, score: Score):
        self.nb_visite += 1
        self.nb_victoire += score
        score = score * -1
        if self.parent != None:
            self.parent.backpropagation(score)

    
# On note que la classe prend toujour le rôle du premier joueur 
# si on veut lui faire jouer le tour du deuième joueur il faut 
# changer la perscpective du plateau
class Alpha_MCTS:
    def __init__(self, game: Gopher, model, hyperparametre: dict) -> None:
        self.game = game
        self.hyperparametre = hyperparametre
        self.model = model
        if self.hyperparametre["type_model"] == "fcnn":
            self.encoder = self.game.encode_flat_state_for_fcnn
        elif self.hyperparametre["type_model"] == "resnet":
            self.encoder = self.game.encode_state_for_convnn
        else:
            raise ValueError("Vous n'avec inclue de model valide dans les hyperparametre")

    @torch.no_grad()
    def get_probs(self, grid: State) -> Action:
        root = alpha_Node(self.game, grid, self.hyperparametre, nb_visite=1 )

        encoded_state = self.encoder(grid)
        
        policy, _ = self.model(encoded_state)
        policy = policy.cpu().numpy()

        # Ajout de bruit pour favoriser l'exploration lors de l'extension de la racine
        policy = (1 - self.hyperparametre['dirichlet_epsilon']) * policy + self.hyperparametre['dirichlet_epsilon'] \
            * np.random.dirichlet([self.hyperparametre['dirichlet_alpha']] * self.game.get_nb_actions())
        legals_bool = self.game.legals_bool(grid, 1)
        policy*=legals_bool
        somme_policy = np.sum(policy)
        if somme_policy == 0.0:
            policy += legals_bool
            somme_policy = np.sum(policy)
        policy /= somme_policy
        root.etendre(policy)

        for i in range(self.hyperparametre['nb_simulation']):
            current_node = root

            # -- SELECTION -- 
            while current_node.is_expand():
                current_node = current_node.select_node()
            
            # on crée un flag pour savoir si on dois etendre et simuler
            expension = True
            # La derniere action est None uniquement pour la root
            if current_node.derniere_action != None:
                # le joueur ayant effectué la dernière action est son parent
                dernier_joueur = current_node.grid[current_node.derniere_action[0]][current_node.derniere_action[1]]
                # On regarde donc si la partie est fini du pdv du noeud actuelle
                final = self.game.final(current_node.grid, -dernier_joueur)
                score  = self.game.score(current_node.grid, -dernier_joueur)

                if final:
                    expension = False

            # -- EXPANSION --

            if expension:
                encoded_state = self.encoder(current_node.grid)
                policy, score = self.model(encoded_state)
                policy = policy.cpu().numpy()
                legals_bool = self.game.legals_bool(current_node.grid, 1)
                policy*=legals_bool
                somme_policy = np.sum(policy)
                if somme_policy == 0.0:
                    policy += legals_bool
                    somme_policy = np.sum(policy)

                policy /= somme_policy
                score = score.item()

                current_node.etendre(policy)
            
            # -- BACKPROPAGATION --
            current_node.backpropagation(score)
        
        nb_case = self.game.get_nb_case()
        action_probs = []
        for i in range(nb_case):
            action_probs.append(0)
        for child in root.children:
            indice = self.game.get_indice_action_in_legal_bool(child.derniere_action)
            action_probs[indice] = child.nb_visite
        tot = sum(action_probs)
        for i in range(len(action_probs)):
            action_probs[i] = action_probs[i]/tot
        # on retourne la liste des action avec leur proba

        return action_probs
        
        
        

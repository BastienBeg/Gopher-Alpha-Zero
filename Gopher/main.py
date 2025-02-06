import os
from game import *
from agents import *
from alpha_MCTS import Alpha_MCTS
from alpha_mcts_trainer import Amcts_trainer
from model_torch import *
import utils as custom_utils
import torch
import time

# Paramètre de torch
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


GRID_TEST_VIDE = ((None, None, None, None, None, 0, 0, 0, 0, 0, 0),
            (None, None, None, None, 0, 0, 0, 0, 0, 0, 0),
            (None, None, None, 0, 0, 0, 0, 0, 0, 0, 0),
            (None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None), 
            (0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
            (0, 0, 0, 0, 0, 0, 0, 0, None, None, None),
            (0, 0, 0, 0, 0, 0, 0, None, None, None, None),
            (0, 0, 0, 0, 0, 0, None, None, None, None, None))

GRID_TEST_1 = ((None, None, None, None, None, 1, -1, 1, -1, 1, -1), 
               (None, None, None, None, 0, 0, 0, 0, 0, 0, 1), 
               (None, None, None, 1, -1, 1, -1, 0, 1, -1, 0), 
               (None, None, -1, 0, 0, 0, 1, -1, 0, 1, -1), 
               (None, 1, 0, 0, 1, -1, 0, 1, -1, 0, 1), 
               (-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0), 
               (1, 0, 0, 1, -1, 0, 0, -1, 0, 0, None), 
               (0, 0, -1, 0, 0, 0, 0, 0, 0, None, None), 
               (0, 0, 0, 0, 0, 0, 0, 0, None, None, None), 
               (0, 0, 0, 0, 0, 0, 0, None, None, None, None), 
               (0, 0, 0, 0, 0, 0, None, None, None, None, None))


GRID_TEST_2 = ((None, None, None, None, None, 0, 0, 0, 0, 0, 0), 
               (None, None, None, None, 0, 0, 0, 0, 0, -1, 0), 
               (None, None, None, 0, 0, 0, 0, 0, 1, 0, 0), 
               (None, None, 0, 0, 0, -1, 1, -1, 0, 0, -1), 
               (None, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1), 
               (0, 0, 0, 0, 0, 0, 0, -1, 1, -1, 0), 
               (0, 0, 0, 0, 0, 0, 1, 0, 0, 1, None), 
               (0, 0, 0, 0, 0, -1, 0, 0, 0, None, None), 
               (0, 0, 0, 0, 0, 1, 0, -1, None, None, None), 
               (0, 0, 0, 0, 0, -1, 1, None, None, None, None), 
               (0, 0, 0, 0, 0, 0, None, None, None, None, None))


hyperparametre_dict= {
    "nb_simulation" : 300, # Nombre d'itération du MCTS
    "UBC_coef_C" : 1.56, # Coef d'exploration du score UBC
    "nb_cycle" : 1, # Nombre de cycle d'entrainement du modèle
    "nb_partie_solo" : 3, # Nombre de partie joué contre lui même lors de chaques cycle
    "epochs" : 3, # Nombre de fois ou le modèle actualise ses poids sur un même jeu de donnée
    "batch_size" : 8, # Taille des paquet de donnée à la fois lors de l'actualisation des poids
    "temperature": 1.70, # Coefficient visant à influencer les choix des coups dans le MCTS (Plus c'est haut, plus c'est random)
    "dirichlet_epsilon": 0.204, # coefficient visant à ajouter du bruit pour le coup juste après la racine dans le MCTS (pour être sur de ne pas passer à côté d'un coups intéressant)
    "dirichlet_alpha": 0.429, # coef de la même formule que le epsilon
    "learning_rate" : 0.001, # Vitesse d'apprentissage des poids
    "weight_decay" :  0.00005, # Diminution des poids pour ne pas stagner sur des mauvais paternes appris
    "nom_model" : "model_resnet-test", # Nom du model, ATTENTION la partie : "model_resnet-" est nécessaire pour l'enregistrement automatique dans le csv
    "type_model" : "resnet", # resnet ou fcnn (implémentation imcomplète pour le fcnn)
    "num_res_blocks" : 4, # nombre de bloc résiduel. AtTTENTION si modification des deux derniers paramètre, il faut modifier l'agent neaural network du fichier agent
    "num_hidden_layers" : 32 # nombre de couche convulotionnelle dans les bloc résiduel (nombre de channel, != nombre de layers)
    }


def main():
    # Mettre à jour le model courant à utiliser dans la stratégie neural network
    custom_utils.update_actual_model("demo_5_min_training\demo_5_min_training_4_cycle.pt")

    debut = time.time()
    game = Gopher(6) # Le chiffre correspond à la taille de la grille
    model_cnn = ResNet(game, 4, 32, device)

    # ------------ Affichage du graphique de la distribution de proba du model ------------
    # with torch.no_grad():
    #     # model_cnn.load_state_dict(torch.load(f = f"models/{get_actual_model()}"))
    #     logit, val = model_cnn(game.encode_state_for_convnn(GRID_TEST_1))
    #     # Logit de base
    #     custom_utils.plot_logit(logit)
    #     # Logit en enlevant les coups illégaux
    #     logit =logit.numpy() * game.legals_bool(GRID_TEST_1, 1)
    #     custom_utils.plot_logit(logit)

    # ------------ Lancement d'optuna pour trouver des hyperparamètres optimaux ------------
    # best_param = custom_utils.optimize_hyperparam_optuna()
    # print(best_param)

    # ------------ Evaluer un model contre un agent aléatoire ------------
    print(custom_utils.winrate_contre_random(game, strategy_neural_network, 20, 10))
    
    # ------------ Entrainer un modele sur les paramèrte du dictionnaire situé au dessus du main ------------
    custom_utils.train_model(game, hyperparametre_dict)
    

    print(f"____Temps : {time.time()-debut}____")

if __name__ == "__main__":
    main()

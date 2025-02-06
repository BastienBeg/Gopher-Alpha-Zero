# import
from game import *
import random
import numpy as np
from alpha_MCTS import *
from model_torch import *
import torch
import numpy as np
import os.path
Strategy = Callable[[Grid, Player, Gopher], Action]
Encoder = Callable[[Grid], list]
Eval_fn = Callable[[Grid, Gopher], float]

def get_actual_model() -> str:
    path_fichier_model_actuel = "models/model_actuel.txt"
    with open(path_fichier_model_actuel, 'r') as file:
        model_path = file.readline().strip()
    return model_path


def get_actual_type_model() -> str:
    """
    fcnn
    resnet
    """
    actual_type = "resnet"
    return actual_type

def get_actual_model_encoder(game: Gopher) -> Encoder:
    
    type_actuel = get_actual_type_model()

    if type_actuel == "fcnn":
        encoder = game.encode_flat_state_for_fcnn
    elif type_actuel == "resnet":
        encoder = game.encode_state_for_convnn
    else:
        raise ValueError("Vous n'avec inclue de model valide dans les hyperparametre")
    return encoder

# - AGENTS -

# -- Agent humain --
def strategy_brain(grid: Grid, player: Player, game: Gopher) -> Action:
    coup_possible = game.legals(grid, player)
    while True:
        for i, coup in enumerate(coup_possible):
            print(f"Coup {i+1} : {coup}")  
        s = input("à vous de jouer, choisissez le numéro du coup: ")
        try:
            t = int(s)
            coup_choisis = coup_possible[t-1]
            return coup_choisis
        except ValueError:
            print("Veuillez entrer un entier valide")
        except IndexError:
            print("selectionnez un entier dans la liste")

# -- Agent simple deterministe --
def strategy_first_legal(grid: State, player: Player, game: Gopher) -> Action:
    playable_moves = game.legals(grid, player)
    if playable_moves != ():
        return playable_moves[0]
    else :
        raise ValueError("Il n'y a plus de coup à jouer")

# -- Agent aléatoire --
def strategy_random(grid: State, player: Player, game: Gopher) -> Action:
    playable_moves = (game.legals(grid, player))
    if playable_moves != ():
       return random.choice(playable_moves)
    else :
        raise ValueError("Il n'y a plus de coup à jouer")


# -- Agent Intelligent Alpha MCTS repose sur des reseau de neurones entrainé en jouant contre lui même via le MCTS --

@torch.no_grad()
def strategy_neural_network(state: State, player: Player, game: Gopher):
    if get_actual_type_model() == "resnet":
        loaded_model = ResNet(game, 4, 32, "cpu")
    elif get_actual_type_model() == "fcnn":
        loaded_model = model_fcnn(game, device)
    else:
        raise ValueError("le type de model n'a pas été correctement implémenté")

    model_path = f"models/{get_actual_model()}"
    loaded_model.load_state_dict(torch.load(f= model_path))
    grid = game.changer_perspective(state, player)

    encoder = get_actual_model_encoder(game)
    encoded_state = encoder(grid)

    action_prob, _ = loaded_model(encoded_state)   
    action_prob = action_prob.cpu().numpy()
    legals_bool = game.legals_bool(grid, 1)
    action_prob+= legals_bool
    action_prob*=legals_bool
    indice_action = np.argmax(action_prob)
    coup = game.get_action_par_indice(indice_action)
    return coup

@torch.no_grad()
def strategy_alpha_mcts(grid: State, player: Player, game: Gopher) -> Action:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if get_actual_type_model() == "resnet":
        loaded_model = ResNet(game, 4, 32, "cpu")
    elif get_actual_type_model() == "fcnn":
        loaded_model = model_fcnn(game, device)
    else:
        raise ValueError("le type de model n'a pas été correctement implémenté")

    loaded_model.load_state_dict(torch.load(f=f"models/{get_actual_model()}"))

    encoder = get_actual_model_encoder(game)
    hyperparametre_dict = {
    "nb_simulation" : 100,
    "UBC_coef_C" : 1.41,
    "nb_cycle" : 4,
    "nb_partie_solo" : 200,
    "epochs" : 3,
    "batch_size" : 16,
    "temperature": 1.25,
    "dirichlet_epsilon": 0.25,
    "dirichlet_alpha": 0.3,
    "learning_rate" : 0.005,
    "weight_decay" :  0.0001,
    "nom_model" : "model_1.0",
    "type_model" : "resnet"
    }
    mcts = Alpha_MCTS(game, loaded_model, hyperparametre_dict)
    grid = game.changer_perspective(grid, player)
    state_probs = mcts.get_probs(grid)
    indice_coup = np.argmax(state_probs)
    list_case = game.list_case()
    coup = list_case[indice_coup]
    return coup


# Statégie pour l'optimisation des hyperparam resnet par optuna

@torch.no_grad()
def strategy_optuna(state: State, player: Player, game: Gopher):
    loaded_model = ResNet(game, 4, 32, "cpu")
    
    # Trouver le dernier model entrainer ainsi que son max cycle
    num_file, i = 0, 1
    while num_file == 0:
        chemin = f"models/model_resnet-{i}/model_resnet-{i}_1_cycle.pt"
        if not os.path.exists(chemin):
            num_file = i-1
        i += 1
    
    num_cycle, j = 0, 1
    while num_cycle == 0:
        chemin = f"models/model_resnet-{num_file}/model_resnet-{num_file}_{j}_cycle.pt"
        if not os.path.exists(chemin):
            num_cycle = j-1
        j += 1

    model_path = f"models/model_resnet-{num_file}/model_resnet-{num_file}_{num_cycle}_cycle.pt"
    loaded_model.load_state_dict(torch.load(f= model_path))
    grid = game.changer_perspective(state, player)

    encoder = get_actual_model_encoder(game)
    encoded_state = encoder(grid)

    action_prob, _ = loaded_model(encoded_state)   
    action_prob = action_prob.cpu().numpy()
    legals_bool = game.legals_bool(grid, 1)
    action_prob+= legals_bool
    action_prob*=legals_bool
    indice_action = np.argmax(action_prob)
    coup = game.get_action_par_indice(indice_action)
    return coup

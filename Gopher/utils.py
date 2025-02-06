from game import *
from alpha_MCTS import Alpha_MCTS
from alpha_mcts_trainer import Amcts_trainer
from model_torch import *
import torch
import time
from agents import *
import csv
import os
import os.path
import optuna

# Prend des logit en entré et plot les logit 
def plot_logit(logit: list[int])-> None:
    # Créer une liste d'indices pour positionner les barres sur l'axe x
    indices = np.arange(len(logit))
    largeur_barre = 0.8
    plt.bar(indices, logit, width=largeur_barre)

    plt.xlabel('Indice des coup')
    plt.ylabel('Prior')
    plt.title('Diagramme de policy')
    plt.show()

# compare une stratégie en jouant des parties contre une strategy random en jouant
# 1/2 le 1er et 1/2 le 2eme
def winrate_contre_random(game: Gopher, 
                          strategy_a_test: Strategy,  # type: ignore
                          nb_game: int = 100, 
                          frequence_affichage:int = 5, 
                          affichage: bool = True) -> float:
    nb_win = 0
    debut = time.time()
    for i in range(nb_game): 

        role = random.randint(1,2)
        if role == 1:
            score = game.jouer_partie(strategy_a_test, strategy_random)
            if score == 1: nb_win += 1
        else:
            score = game.jouer_partie(strategy_random, strategy_a_test)
            if score == -1: nb_win += 1


        if (1+i)%frequence_affichage == 0 and affichage:
            print(f"{nb_win} / {i+1} parties")

    winrate = nb_win/nb_game*100
    winrate = round(winrate, 4)
    if affichage:
        print(f"La stratégie à gagné {nb_win} partie sur {nb_game} parties joué avec un winrate final de {winrate}% \nPour un temps d'eecution de {(time.time() - debut):.2f} secondes")
    return winrate

# Entraine un model selon des hyperparamètres et enregistre tout les models
def train_model(game : Gopher, hyperparametre: dict,):
    model = ResNet(game, hyperparametre["num_res_blocks"], hyperparametre["num_hidden_layers"], device)
    a_mcts = Alpha_MCTS(game, model, hyperparametre)

    learning_r = hyperparametre["learning_rate"]
    weight_d = hyperparametre["weight_decay"]
    nom_model = hyperparametre["nom_model"]

    optim = torch.optim.Adam(model.parameters(), lr= learning_r, weight_decay= weight_d)
    trainer = Amcts_trainer(game, model, a_mcts, optim, hyperparametre)

    # Calcul du temps d'aprentissage
    temps_full_cycle = time.time()
    trainer.aprentissage(nom_model)
    temps_full_cycle = round((time.time() - temps_full_cycle), 1)

    nb_cycle = hyperparametre["nb_cycle"]
    for i in range(nb_cycle):
         with open("models/_info.csv", mode='a', newline='') as csvfile:
            fieldnames = ['nom_model', 'type_reseau_de_neurone', 'winrate', 'training_time', 'nb_partie', 'nb_epoch', 'nb_simulation', \
                          'batch_size', 'UBC_coef_C', 'temperature', 'dirichlet_epsilon', 'dirichlet_alpha', 'learning_rate', 'weight_decay']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Calcul du winrate :
            model_path = convert_to_model_path( f"{hyperparametre["nom_model"]}_{i+1}_cycle.pt")
            update_actual_model(model_path)
            winrate = winrate_contre_random(game, strategy_neural_network, 200, 300)

            # Écrire une nouvelle ligne avec les informations fournies
            writer.writerow({'nom_model': f"{hyperparametre["nom_model"]}_{i+1}_cycle.pt",
                            'type_reseau_de_neurone': hyperparametre["type_model"],
                            'winrate': winrate,
                            'training_time': temps_full_cycle/nb_cycle,
                            'nb_partie': hyperparametre["nb_partie_solo"],
                            'nb_epoch': hyperparametre["epochs"],
                            'nb_simulation': hyperparametre["nb_simulation"],
                            'batch_size' : hyperparametre["batch_size"],
                            'UBC_coef_C' : hyperparametre["UBC_coef_C"],
                            'temperature': hyperparametre["temperature"],
                            'dirichlet_epsilon': hyperparametre["dirichlet_epsilon"],
                            'dirichlet_alpha': hyperparametre["dirichlet_alpha"],
                            'learning_rate' : hyperparametre["learning_rate"],
                            'weight_decay' :  hyperparametre["weight_decay"],
                            })

# Prend 2 stratégies, les fais jouer l'un contre l'autre et renvoie le winrate
def compare_strategy(game: Gopher, 
                          strat_1,
                          strat_2,
                          nb_game: int = 100, 
                          frequence_affichage:int = 5, 
                          affichage: bool = True) -> float:
    nb_win = 0
    debut = time.time()
    for i in range(nb_game): 
        score = game.jouer_partie(strat_1, strat_2)
        if score == 1: nb_win += 1

        if (1+i)%frequence_affichage == 0 and affichage:
            print(f"{nb_win} / {i+1} parties")
    winrate = nb_win/nb_game*100
    if affichage:
        print(f"La stratégie à gagné {nb_win} partie sur {nb_game} parties joué avec un winrate final de {winrate}% \nPour un temps d'eecution de {(time.time() - debut):.2f} secondes")
    return winrate



# Optimisation des hyperparamètre du model ResNet avec optuna
def optimize_hyperparam_optuna(): 

    def objective(trial):
        # On définie les hyperparamètres à optimiser
        nb_simulation = trial.suggest_int('nb_simulation', 100, 400)
        UBC_coef_C = trial.suggest_float('UBC_coef_C', 1.0, 2.0)
        nb_cycle = trial.suggest_int('nb_cycle', 4, 15)
        nb_partie_solo =  trial.suggest_int('nb_partie_solo', 8, 40)
        nb_epoch = trial.suggest_int('nb_epoch', 1, 7)
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
        temperature = trial.suggest_float('temperature', 0.5, 2.0)
        dirichlet_epsilon = trial.suggest_float('dirichlet_epsilon', 0.1, 0.5)
        dirichlet_alpha = trial.suggest_float('dirichlet_alpha', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        # Création du dictionnaire d'hyperparamètre
        num_file, i = 0, 1
        while num_file == 0:
            chemin = f"models/model_resnet-{i}/model_resnet-{i}_1_cycle.pt"
            if not os.path.exists(chemin):
                num_file = i
            i += 1

        hyper_optuna_dict = {
        "nb_simulation" : nb_simulation,
        "UBC_coef_C" : UBC_coef_C,
        "nb_cycle" : nb_cycle,
        "nb_partie_solo" : nb_partie_solo,
        "epochs" : nb_epoch,
        "batch_size" : batch_size,
        "temperature": temperature,
        "dirichlet_epsilon": dirichlet_epsilon,
        "dirichlet_alpha": dirichlet_alpha,
        "learning_rate" : learning_rate,
        "weight_decay" :  weight_decay,
        "nom_model" : f"model_resnet-{num_file}",
        "type_model" : "resnet",
        "num_res_blocks" : 4,
        "num_hidden_layers" : 32
        }

        # Entrainement du model avec les paramètres suggérés par optuna
        game = Gopher(6)
        model_resnet = ResNet(game, 4, 32, device)
        train_model(game, model_resnet, hyper_optuna_dict)

        # On calcul le winrate qui sert de métrique de performance
        winrate = winrate_contre_random(game, strategy_optuna, 600, 1000)
        return winrate

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    return best_params


# donne le chemin relatif complet
def convert_to_relative_path(filename):
    parts = filename.split('_')
    model_name = parts[0]  
    model_num = parts[1]  
    relative_path = f"models/{model_name}_{model_num}/{filename}"
    return relative_path


# Donne le chemin pour actualiser le model courant
def convert_to_model_path(filename):
    parts = filename.split('_')
    model_name = parts[0]   
    model_num = parts[1]  
    relative_path = f"{model_name}_{model_num}/{filename}"
    return relative_path

def update_actual_model(model_path):
    path_fichier_model_actuel = "models/model_actuel.txt"
    with open(path_fichier_model_actuel, 'w') as file:
        file.write(model_path + '\n')

# Parcours le csv et calcul le winrate de tout les models qui n'en n'ont pas
def update_winrate(csv_file, taille_plateau):
    with open(csv_file, 'r+', newline='') as file:
        reader = csv.DictReader(file)
        writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
        rows = list(reader)
        updated_rows = []

        for row in rows:
            if not row['winrate']: 
                nom_model = row['nom_model']
                relative_path = convert_to_relative_path(nom_model)
                path_to_update = convert_to_model_path(nom_model)
                if os.path.exists(relative_path):
                    update_actual_model(path_to_update)
                    game = Gopher(taille_plateau)
                    winrate = winrate_contre_random(game, strategy_neural_network, 500, 600)
                    row['winrate'] = winrate
                    print(f"Le model {nom_model} à était actualisé à un winrate de {winrate}")
            updated_rows.append(row)

        file.seek(0)
        writer.writeheader()
        writer.writerows(updated_rows)
        file.truncate()

def plot_probs_and_legals_prob(game: Gopher, state: State):
    pass

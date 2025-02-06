# - Import -

from pathlib import Path
from game import * 
import random
import numpy as np
import torch
from alpha_MCTS import *
from model_torch import *

# Paramètre de torch
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dico d'hyperparamètre :
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
class Amcts_trainer:
    def __init__(self,
                game : Gopher, 
                model: model_fcnn, 
                a_mcts : Alpha_MCTS, 
                optimizer, 
                hyperparametre: dict):
        self.game = game
        self.model = model
        self.alpha_mcts = a_mcts
        self.optimizer = optimizer
        self.nb_boucle_apprentissage = hyperparametre["nb_cycle"]
        self.nb_partie = hyperparametre["nb_partie_solo"] 
        self.nb_epoch =  hyperparametre["epochs"]
        self.batch_size =  hyperparametre["batch_size"]
        self.temperature = hyperparametre["temperature"]
        self.type_model = hyperparametre["type_model"]

        
    def jouer_partie(self) -> list[tuple[State, list, Score]]:
        """
        Fait des parties contre lui même et les ajoute à un historique
        pour pouvoir entrainer le reseau de neurone dessus
        """
        memoire = []
        joueur = 1
        state = self.game.create_grid()
        
        while not self.game.final(state, joueur):
            neutral_state = self.game.changer_perspective(state, joueur)
            action_probs = self.alpha_mcts.get_probs(neutral_state)
            action_probs = np.array(action_probs)
            memoire.append((neutral_state, action_probs, joueur))
            
            temperature_effective = action_probs ** (1/ self.temperature)
            temperature_effective = temperature_effective / np.sum(temperature_effective)
            # temperature effective est proche de 1 mais pas =, peut etre source d'erreur

            action = np.random.choice(self.game.get_nb_actions(), p=temperature_effective)
            action = self.game.get_action_par_indice(action)
            
            # On fait jouer le joueur actuel car cet état est constant au fil des itération. neutral state != state
            state = self.game.play(state, action, joueur)
            
            joueur = self.game.get_ennemie(joueur)

        score = self.game.score(state, joueur)
    
        return_memoire = []
        for hist_neutral_state, hist_action_probs, hist_joueur in memoire:
            hist_result = score if hist_joueur == joueur else -score
            # On encode le state pour le model
            if self.type_model == "fcnn":
                hist_encoded_state = self.game.encode_flat_state_for_fcnn(hist_neutral_state)
            elif self.type_model == "resnet":
                hist_encoded_state = self.game.encode_state_for_convnn(hist_neutral_state)
            else:
                raise ValueError("Vous n'avec inclue de model valide dans les hyperparametre")
            
            return_memoire.append((hist_encoded_state, hist_action_probs, hist_result))
        return return_memoire
    
    def train(self, memory):
        """
        Entraine le reseau de neurone sur un historique de parties
        """
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.batch_size):
            if batchIdx == len(memory) -1:
                safeBatchIdx = batchIdx-1
            else:
                safeBatchIdx = batchIdx
                
            sample = memory[safeBatchIdx:min(len(memory) - 1, safeBatchIdx + self.batch_size)]
            state, policy_targets, value_targets = zip(*sample)

            value_targets =  np.array(value_targets).reshape(-1, 1)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_targets = torch.tensor(value_targets, dtype=torch.float32)

            

            # 1.Forward pass
            out_policy, out_value = self.model(state)
            
            # 2.Calcul de la loss
            policy_targets = torch.tensor(policy_targets)

            #   Si à la fin du batch il ne reste q'un éléments, on le met dans la bonne dimension
            if out_policy.dim() == 1:
                out_policy = torch.unsqueeze(out_policy, dim = 0)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            # 3. Reset des gradients
            self.optimizer.zero_grad()

            # 4. Calcul des gradients associé au fonctions de loss (crossEntropy et MSE dans notre cas)
            loss.backward()

            # 5. On met à jour les poids du model en fonction de la fonction d'optimisation (ex Adam, SGD, RMSprop (dans notre cas Adam))
            self.optimizer.step() 
 
    def aprentissage(self, nom_model = "model_tmp", nom_optim = "optim_tmp"):
        """
        Fait un certains nombre de cycle d'alternance entre jouer contre sois même pour enregistrer des parties
        et s'entrainer sur ces parties, et enfin il enregistre les poids du modele
        """
        # boucle de cycle d'apprentissage
        for i in range(self.nb_boucle_apprentissage):
            print(f"Boucle d'apprentissage num : {i+1}")
            memory = []
            
            self.model.eval()
            # nombre de partie effectué contre lui même
            for nb_partie in range(self.nb_partie):
                if nb_partie%1 == 0:
                    print(f"    Engristrement de la partie : {nb_partie+1}")
                memory += self.jouer_partie()
                
            self.model.train()
            # nombre de fois où le model s'entraine sur les données des parties faites
            for epoch in range(self.nb_epoch):
                # print(f"    Entrainement numéro : {epoch+1} sur les parties")
                self.train(memory)
            
            # Create models directory (if it doesn't already exist)
            MODEL_DIR_PATH = Path("models")
            MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True)

            MODEL_SOUS_DIR_NAME = f"{nom_model}"
            MODEL_SOUS_DIR_PATH = MODEL_DIR_PATH / MODEL_SOUS_DIR_NAME
            MODEL_SOUS_DIR_PATH.mkdir(parents=True, exist_ok=True)

            # Create model save path
            MODEL_NAME = f"{nom_model}_{i+1}_cycle.pt"
            OPTIM_NAME = f"{nom_optim}_{i+1}_cycle.pt"
            MODEL_SAVE_PATH = MODEL_SOUS_DIR_PATH / MODEL_NAME
            OPTIM_SAVE_PATH = MODEL_SOUS_DIR_PATH / OPTIM_NAME

            # Save the model state dict
            print(f"Saving model to: {MODEL_SAVE_PATH}")
            torch.save(obj=self.model.state_dict(), f=MODEL_SAVE_PATH)
            torch.save(self.optimizer.state_dict(), f=OPTIM_SAVE_PATH)

            """
            A noté, pour load le model, la synthaxe est la suivante lors de l'initialistation du model (dans le main ou agent) :  
            loaded_model.load_state_dict(torch.load(f=nom_du_model.pth))
            """
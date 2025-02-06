import numpy as np 
from game import *
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Modele dense
class model_fcnn(nn.Module):

    def __init__(self, game : Gopher, device):

        super(model_fcnn, self).__init__()

        self.device = device
        self.game = game
        self.nb_actions = self.game.get_nb_actions()
        self.nb_case = self.game.get_nb_case()

        self.fc1 = nn.Linear(in_features=self.nb_case, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)

        # On a besoin de deux sorties, une pour les valeur du noeud et une pour les probas des actions
        self.action_head = nn.Linear(in_features=128, out_features=self.nb_actions)
        self.value_head = nn.Linear(in_features=128, out_features=1)

        self.to(device)

    def forward(self, x): 
        x = np.array(x)
        x = torch.tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_logits = self.action_head(x)
        value_logit = self.value_head(x)

        return F.softmax(action_logits, dim=0), torch.tanh(value_logit)
    
    def get_coup_value(self, state: State):
        action_prob, value = self.forward(state)
        indice_coup = torch.argmax(action_prob)
        list_case = self.game.list_case()
        coup = list_case[indice_coup]
        value = value.item()
        return coup, value
    
# Modele Convolutionnelle et résiduel
class ResNet(nn.Module):
    def __init__(self, game: Gopher, num_resBlocks: int, num_hidden : int, device: str):
        super().__init__()
        self.game = game
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.get_nb_case_including_none(), game.get_nb_actions())
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.get_nb_case_including_none(), 1),
            nn.Tanh()
        )
        
        self.to(device)
        
    def forward(self, x):
        x = torch.tensor(x)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        policy = torch.squeeze(policy)
        return F.softmax(policy, 0), torch.tanh(value)
        
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = F.relu(x)
        return x



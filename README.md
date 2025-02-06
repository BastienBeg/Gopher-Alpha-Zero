# Intro

Poussé par la curiosité, je me suis lancé dans une tentative d'implémentation de l'algorithme Alpha-Zero pour concevoir un agent intelligent pour le jeu Gopher.
Ce jeu a été créé par Mark Steere, les règles sont disponibles dans le pdf du repo "Gopher_hex_rules.pdf".  

Je tiens à préciser, avant tout propos, que la totalité des sources que j'ai pu utiliser dans ce projet sont incluses à la fin de ce document texte dans la section : "Ressources"

Pour vous présenter ce projet, je vais tout d'abord vous présenter l'architecture du modèle ainsi que sa boucle d'entraînement, puis ensuite, je vous présenterai mes tentatives d'entraînement et enfin, les résultats que j'ai obtenus.

# Architecture et Implémentation du modèle

## Réseau de neurones

Premièrement, le réseau de neurones est implémenté avec pytorch dans le fichier "model_torch.py". J'ai testé deux architectures :
- Une architecture dense peu profonde. Je lui donne donc en entrée une grille aplatit du jeu représenté par 3 nombres possible, et il sort une distribution de probabilité correspondant à sa confiance pour chaque coup, qu'il soit le meilleur, ainsi qu'une évaluation de l'état. Néanmoins, cette architecture est très limitée du fait qu'il est difficile pour le modèle d'encoder des paternes depuis une grille plate.
- Une architecture convolutionelle et résiduelle. Elle traite donc la grille de jeu comme une image en prenant en entrée une grille sous forme d'image possédant 3 couches de booléens représentant les positions des deux joueurs ainsi que les cases libres. Elle possède également un nombre variable de blocs résiduels et de couches convolutionnelles. Elle possède les même outputs que l'architecture précédente 

## Alpha MCTS

Afin d'entraîner le modèle, on fait donc appel à l'architecture d'Alpha-Zero. Nous avons donc un fichier "alpha_MCTS.py" qui contient la classe permettant d'instancier un MCTS. Ce MCTS à la particularité de prendre un réseau de neurones en paramètre qui va lui permettre d'attribuer des scores de "priorité" de visite pour les différents nœuds de l'arbre lors de la sélection, et en particulier lors du calcul du score UBC. On va également remplacer les simulations (méthode utilisée dans un MCTS classique) par la fonction d'évaluation de notre modèle (on rappelle que le modèle a deux têtes de sorties). 
Enfin, notre MCTS ne nous donne pas un coup à jouer, mais une distribution de probabilité similaire à celle de notre modèle, au détail près qu'elle a été affiné par l'exploration de l'arbre.

## Trainer

Pour entraîner notre modèle, nous avons le fichier "alpha_mcts_trainer.py" qui contient la définition de notre classe nous permettant d'entraîner notre modèle.
Nous pouvons distinguer deux parties dans un cycle d'apprentissage, que nous répéterons en fonction du nombre de cycles souhaité.
1. On va faire s'affronter notre modèle contre lui-même au travers du MCTS. Pour chaque partie nous prendrons soin de sauvegarder l'historique des distributions de probabilité du MCTS, ainsi que le coup joué et de la victoire ou non du modèle dans la partie correspondante.
2. Ensuite, nous allons entraîner le réseau de neurones sur ces données en calculant la perte par la différence entre les deux distributions de probabilités ainsi que la différence entre l'évaluation de la grille et le résultat final obtenue. Une fois ses poids actualisés, on peut repartir pour un nouveau cycle.

# Résultat

Pour visualiser les résultats, toutes les infos concernant les modèles se trouvent dans le csv situé dans le dossier "models/".

Il est difficile d'interpréter les résultats avec des temps d'entraînement aussi court et un nombre de parties trop peu important pour avoir un échantillon de coup significatif. Néanmoins, on observe tout de même un niveau significativement meilleur que l'aléatoire sur certains modèles. En analysant les différents paramètres et modèles du csv, j'ai entraîné un modèle pendant 5 min avec les hyperparamètres actuellement chargé dans le fichier "main.py". Ce modèle atteint environ 90% de réussite contre un joueur aléatoire (moyenne faite sur 500 parties). Pour cette échelle d'entraînement et de temps de calcul, ce résultat est plutôt satisfaisant même s'il ne donne pas de garantie sur la convergence pour un entraînement plus long.

# Comment utiliser le programme

Pour entraîner un modèle, dans le fichier main.py, vous trouverez un dictionnaire possédant tous les hyperparamètres ajustables, avec un commentaire précisant leur utilité. Ensuite, dans la fonction main, vous devriez trouver les lignes avec des commentaires explicites pour lancer l'entrainement et tester les modèles. Si vous avez des questions supplémentaires sur l'utilisation du programme, n'hésitez pas à m'envoyer un mail à l'adresse suivante : "bastienbeghin@gmail.com".

nb : il faut lancer le programme depuis le dossier Gopher.

# Axes d'amélioration

J'ai identifié certains des points du projet pouvant être améliorés :

- **L'optimisation du programme du jeu.** En effet la génération de parties lorsque le programme joue contre lui-même est très longue. Les performances du jeu pourraient être améliorées en programmant le jeu sur un langage plus rapide en faisant appel à des microservices de python pour garder la partie réseau de neurones dessus. Il serait par exemple envisageable d'utiliser le GoLang pour paralléliser les parties avec du multi-threading.

- **Entrainement plus approfondie.** Pour des raisons de puissance de calcul et de temps disponible, je n'ai pas tenté d'entraînement sur des périodes très longues, et avec un grand nombre de parties par cycle.

- **Recherche de meilleurs hyperparamètres**. J'ai utilisé optuna pour trouver des paramètres satisfaisant, néanmoins pour des raisons de calcul, je n'ai pas pu le lancer durant des périodes assez longues.

Il reste sûrement d'autres améliorations possibles, cette liste évoluera au fil des retours.

# Ressources

Liens Réseau de neurones / Pytorch:
*NN from scratch (Samson Zhang)*
- https://www.youtube.com/watch?v=w8yWXqWQYmU&t=878s

*Daniel Bourke*
- https://www.learnpytorch.io/
- https://www.youtube.com/watch?v=Z_ikDlimN6A

*Convolutional nn*
- https://poloclub.github.io/cnn-explainer/

*Resnet*
- https://www.youtube.com/watch?v=o_3mboe1jYI&pp=ygUdcmVzbmV0IGFyY2hpdGVjdHVyZSBleHBsYWluZWQ%3D
- https://www.youtube.com/watch?v=Q1JCrG1bJ-A&pp=ygUdcmVzbmV0IGFyY2hpdGVjdHVyZSBleHBsYWluZWQ%3D

Liens MCTS / Alpha MCTS :
*DeepMind AlphaZero*
- https://arxiv.org/pdf/1712.01815

*General alpha zero implementation*
- https://github.com/suragnair/alpha-zero-general/blob/master/README.md

*freeCodeCamp.org*
- https://www.youtube.com/watch?v=wuSQpLinRB4&t=626s

*Kevin Lubick*
- https://www.youtube.com/watch?v=c8SLNEpFSrs

*Josh Varty*
- https://www.youtube.com/watch?v=62nq4Zsn8vc
- https://joshvarty.github.io/AlphaZero/

*Tim Miller, The University of Queensland*
- https://gibberblot.github.io/rl-notes/single-agent/mcts.html

*Ankit Choudhary*
- https://www.analyticsvidhya.com/blog/2018/11/reinforcement-learning-introduction-monte-carlo-learning-openai-gym/

Liens NEAT algorithme
*Dama paper with Neat*
- https://www.informatica.si/index.php/informatica/article/view/3897/1698

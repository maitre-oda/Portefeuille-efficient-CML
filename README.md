# Portefeuille-efficient-CML
Analyse d'un portefeuille à deux actifs — Programme interactif
=================================================================

Ce projet permet d'analyser un portefeuille composé de deux actifs financiers
choisis par l'utilisateur via le terminal. Le programme télécharge les données
de marché, calcule les rendements, affiche des statistiques, trace la frontière
efficiente et calcule le portefeuille optimal (max Sharpe).

1) Installation des dépendances
-------------------------------
Installez les bibliothèques nécessaires :

    pip install -r requirements.txt

ou :

    pip install numpy pandas yfinance matplotlib scipy

2) Lancement du programme
-------------------------
Dans un terminal :

    python analyse_portefeuille.py

Le programme affichera une liste d'environ 50 actions (USA + Europe).
Saisissez les indices de deux actions à analyser, puis entrez :
- le taux sans risque annuel (ex : 0.02 pour 2%)
- éventuellement les dates de début/fin

3) Résultats fournis
--------------------
Le script produit :
- Statistiques descriptives
- Courbes des rendements log quotidiens
- Volatilités et corrélation glissantes
- Frontière efficiente (sampling)
- Portefeuille de Sharpe maximum
- CML (Capital Market Line)
- Fichiers CSV des séries téléchargées

4) Fichiers produits automatiquement
------------------------------------
- prices_two_assets.csv
- log_returns.csv

5) Auteur
---------
Programme réalisé pour analyse académique.


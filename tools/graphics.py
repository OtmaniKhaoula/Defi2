# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:47:27 2023

@author: Khaoula Otmani & Elisa Martin
"""

import matplotlib.pyplot as plt
from nltk import word_tokenize
import pandas as pd 

# Répartition des notes (données apprentissage)
def graph_repartition(distrib_notes, titre):
    plt.bar(range(10), list(distrib_notes.values()), width = 0.6, color = 'red',  edgecolor = 'black', linewidth = 2,  ecolor = 'magenta', capsize = 10)
    plt.xticks(range(10), list(distrib_notes.keys()), rotation = 45)
    plt.title("Répartition des " + titre)
    plt.xlabel("Notes")
    plt.ylabel("Effectifs")
    plt.savefig(f"../graphs/{titre}.png")

# Répartition des notes par films (n: nombre de film qu'on veut mettre dans le graphique)
# titre: préciser si c'est par films ou par utilisateur et si c'est sur les données d'apprentissage ou de validation
def graph_repartition_by(distrib_notes, n, titre):
    
    # Récupérer les films qu'on veut intégrer dans le graphique 
    films = list(distrib_notes.keys())[0:4]
    #print("films = ", films)
    #print(distrib_notes[films[0]])
    
    barWidth = 0.075
    colors = ['#00FFA1', '#00FFD0', '#00FAFF', '#00E1FF', '#00AEFF', '#0098FF', '#007FFF', '#0065FF', '#004CFF', '#3100CD']

    plt.figure(figsize=(15,9))
    for i in range(10):
        # distrib_notes_by_movie.iloc[0:4, i]
        y = [distrib_notes[key][i][0] for key in films]
        r = [x + barWidth*i for x in range(len(y))]
        plt.bar(r, y, width = barWidth, color = [colors[i] for j in y], linewidth = 2, label = 0.5+i*0.5)
        plt.title("Répartition des notes pour les 4 premiers" + titre)
        plt.xlabel("Films")
        plt.ylabel("Effectifs")

    plt.xticks([r + barWidth*5 for r in range(len(y))], [key for key in films])
    plt.title('Notes')
    plt.savefig(f"../graphs/{titre}.png")
    
# Récupérer le nombre de commentaires par films (nombre de lignes associées à chaque films)

# Récupérer le nombre de notes mis par utilisateurs (nombre de lignes par utilisateurs)

        
# Nuage de word_cloud
        
            


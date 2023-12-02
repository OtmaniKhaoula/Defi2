# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:47:27 2023

@author: Khaoula Otmani & Elisa Martin
"""

import matplotlib.pyplot as plt
import pandas as pd
from nltk import word_tokenize
from wordcloud import WordCloud
import plotly.express as px
import sys, os

plt.style.use('ggplot')

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)

import config
#from wordcloud import WordCloud 

path = config.paths['url']

# Répartition des notes (données apprentissage)
def graph_repartition(distrib_notes, titre, xLabel):
    plt.figure(figsize=(12,8))
    plt.bar(range(10), list(distrib_notes.values()), width = 0.6, color = 'red',  edgecolor = 'black', linewidth = 2,  ecolor = 'magenta', capsize = 10)
    plt.xticks(range(10), list(distrib_notes.keys()), rotation = 45)
    plt.title(titre)
    plt.gca().set_facecolor('white')
    plt.xlabel(xLabel)
    plt.ylabel("Effectifs")
    plt.savefig(f"{path}/graphs/{titre}.png")
    plt.clf()

# Comparer la répartition des notes pour les données d'apprentissage
def graph_note_repartition(distrib_notes_app, distrib_notes_dev, titre, name1, name2, xLabel):

    barWidth = 0.3
    
    # Récupérer les films qu'on veut intégrer dans le graphique (les y) 
    films_app = list(distrib_notes_app.values())
    films_dev = list(distrib_notes_dev.values())
    
    #print("films = ", films)
    #print(distrib_notes[films[0]])
    x1 = range(len(films_app)) # Position des barres des données d'app
    x2 = [i + barWidth for i in x1] # des données de dev
    
    plt.figure(figsize=(12,8))
    plt.bar(x1, films_app, width = barWidth, color = 'orange', linewidth = 2, label = name1)
    plt.bar(x2, films_dev, width = barWidth, color = 'yellow', linewidth = 2, label = name2)

    plt.xticks([r + barWidth / 2 for r in range(len(films_app))], list(distrib_notes_app.keys()))

    plt.title(titre)
    plt.gca().set_facecolor('white')
    plt.xlabel(xLabel)
    plt.ylabel("Effectifs")
    
    plt.legend(title = 'Donnée',  facecolor='white', edgecolor='black')

    #plt.show()
    plt.savefig(f"{path}/graphs/{titre}.png")

    #plt.clf()
        
# Répartition des notes par films (n: nombre de film qu'on veut mettre dans le graphique)
# titre: préciser si c'est par films ou par utilisateur et si c'est sur les données d'apprentissage ou de validation
def graph_repartition_by(distrib_notes, n, titre):
    
    # Récupérer les films qu'on veut intégrer dans le graphique 
    films = list(distrib_notes.keys())[0:n]
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

    plt.gca().set_facecolor('white')
    plt.xticks([r + barWidth*5 for r in range(len(y))], [key for key in films])
    plt.legend(title = 'Notes')

    plt.savefig(f"{path}/graphs/{titre}.png")
    plt.clf()

"""
def graph_boxplot(distrib_notes, tit, titre):

    plt.boxplot(distrib_notes)
    
    plt.title(tit)
    plt.ylim(-0.5, 6)    
    plt.savefig(f"{path}/graphs/{titre}.png")
    plt.clf()
"""
  
def graph_boxplot(distrib_notes, n, name):
    # Récupérer les films qu'on veut intégrer dans le graphique 
    keys = list(distrib_notes.keys())[0:n]
    notes = []
    for key in keys:
        notes.append([distrib_notes[key][i][1] for i in range(len(distrib_notes[key])) for _ in range(distrib_notes[key][i][0])])

    fig = plt.figure(figsize =(12, 9))
 
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    
    # x-axis labels
    ax.set_xticklabels(keys, rotation=45, ha='right')
 
    # Creating plot
    ax.boxplot(notes)
    ax.set_xlabel("Identifiants des " + name)
    ax.set_ylabel("Notes")
    
    # Adding title 
    plt.title("Dispersion des notes")
    plt.gca().set_facecolor('white')
 
    titre = "Dispersion des notes pour les " + name
    plt.savefig(f"{path}/graphs/{titre}.png")

# Nuage de word_cloud
def word_cloud(grade_words, titre):
    
    # Création d'une nouvelle structure de données avec uniquement les mots et les fréquences
    new_grade_words = {key: value[2] for key, value in grade_words.items()}

    # Création de l'objet WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(new_grade_words)
    
    # Affichage du nuage de mots
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Désactiver les axes
    plt.savefig(f"{path}/graphs/{titre}.png")


# Boxplot entre les notes et des utilisateurs ou des films 
# Boxplot pour les mots les plus fréquents associés aux notes

   
# descriptif pour tous les films , Nombre utilisateur et films
# Corrélation (films/notes   utilisateur/notes)
# Longueur des phrases (par char et par mot)
# mot les plus frequents par notes
# Dictionnaire avec commentaire comme clé et la liste des notes en valeurs 
# correlation entre la note / longueur commentaire (ou autre)
# note moyenne pour chaque mot fréquent 
# Corrélation note - mot fréquents 

# Notes selon url 




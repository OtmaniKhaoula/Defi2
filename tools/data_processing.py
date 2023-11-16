# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:27:23 2023

@author: Khaoula Otmani & Elisa Martin
"""

# chargement des librairies 
import pandas as pd
import nltk
import string
import spacy
import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords

#Supprimer les mots d'arrêts classiques en Français
stopwords = stopwords.words("french")

# Exemple
#text = "Le traitement, + du langage naturel permet aux ordinateurs de comprendre le langage naturel comme le font les humains. Que la langue soit parlée ou écrite, le traitement du langage naturel utilise l’intelligence artificielle pour prendre des données du monde réel, les traiter et leur donner un sens qu’un ordinateur peut comprendre."

# Nettoyer le texte
#Chargement des lemmes de la langue française
nlp = spacy.load('fr_core_news_md')

# dict_comments: Dictionnaire avec identifiant comme clé et text comme valeur
def preprocessing_text(dict_comments):
    new_dict_comments = {}
    for key, text in tqdm.tqdm(dict_comments.items()):
        if(text == None):
            new_dict_comments[key] = ""
            continue
        
        #Tokenization
        words = word_tokenize(text,language="french",preserve_line=True)

        #Lemmatisation
        #words=nlp(" ".join(words))

        #Création d'une liste vide pour aceullir les mots sans ponctutation
        clean_words = []

        #Enlever la ponctuation et les stopwords:
        for w in words:
            if str(w).isalpha() and w not in stopwords:
                clean_words.append(w.lower())

        new_dict_comments[key] = clean_words
    
    return new_dict_comments




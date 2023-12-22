# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:27:23 2023

@author: Khaoula Otmani & Elisa Martin
"""

# chargement des librairies 
import numpy as np
import nltk
import string
import spacy
import tqdm
import csv
from nltk import word_tokenize
from nltk.corpus import stopwords
import sys, os
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)

import config
path = config.paths['url']

#Supprimer les mots d'arrêts classiques en Français
stopwords = stopwords.words("french")

stopwords.append('d')
stopwords.append('l')
stopwords.append("un")
stopwords.append("le")
stopwords.append('n')
stopwords.append("une")
stopwords.append("la")
stopwords.append('c')
stopwords.append('a')

# Nettoyer le texte
#Chargement des lemmes de la langue française
nlp = spacy.load('fr_core_news_md')
 
# dict_comments: Dictionnaire avec identifiant comme clé et text comme valeur
def preprocessing_text(dict_comments):
    new_dict_comments = {}
    for key, text in dict_comments.items():
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
            w = w.lower()
            if str(w).isalpha() and w not in stopwords:
                clean_words.append(w)

        new_dict_comments[key] = clean_words
    
    return new_dict_comments

def preprocessing_fasttext(dict_comments, notes, users_id, movies_id, folder):
    training_data = []
    n = {}

    for i in np.arange(0.5, 5.5, 0.5):
        n[i] = 0
    
    with open(f'{path}/processed_data/{folder}/data.tsv', 'w', newline='') as f:
        for key, text in tqdm.tqdm(dict_comments.items()):
            if(text == None):
                training_data.append([key, ""])
                continue
            
            #Tokenization
            words = word_tokenize(text,language="french",preserve_line=True)

            #Création d'une liste vide pour aceullir les mots sans ponctutation
            clean_words = []

            #Enlever la ponctuation et les stopwords:
            for w in words:
                w = w.lower()

                if str(w).isalpha() and w not in stopwords:
                    #clean_words.append(w.lemma_)
                    clean_words.append(w)


            txt = " ".join(clean_words)
            if folder == "test":
                row = key+" "+txt
            else:
                row = "__label__"+str(notes[key])+" "+txt #+" "+str(users_id[key])+" "+str(movies_id[key])

            output = csv.writer(f, delimiter='\t')
            output.writerow([row])

            n[notes[key]] += 1
    
def preprocessing_test(dict_comments):
    new_dict_comments = {}
    for key, text in dict_comments.items():
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
            w = w.lower()
            if str(w).isalpha() and w not in stopwords:
                clean_words.append(w)

        new_dict_comments[key] = " ".join(clean_words)
    
    return new_dict_comments
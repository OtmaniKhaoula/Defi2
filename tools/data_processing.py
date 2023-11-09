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
from nltk import word_tokenize
from nltk.corpus import stopwords

# Exemple
text = ["Le traitement, + du langage naturel permet aux ordinateurs de comprendre le langage naturel comme le font les humains. Que la langue soit parlée ou écrite, le traitement du langage naturel utilise l’intelligence artificielle pour prendre des données du monde réel, les traiter et leur donner un sens qu’un ordinateur peut comprendre."]

# Nettoyer le texte
#Chargement des lemmes de la langue française
nlp = spacy.load('fr_core_news_md')

#Tokenization
words = word_tokenize(text,language="french",preserve_line=True)

#création d'une liste vide pour aceullir les mots sans ponctutation
words_no_punc = []

#Enlever la ponctuation :
for w in words:
    if w.isalpha():
        words_no_punc.append(w.lower())

#Supprimer les mots d'arrêts classiques en Français
stopwords = stopwords.words("french")

#Liste vide pour stocker les mots nétoyés :
clean_words = []

#Remplissage de la liste avec les mots nétoyés
for w in words_no_punc:
    if w not in stopwords:
        clean_words.append(w)

#liste vide pour aceuillir les mots lematisés
clean_words_lem = []

#remplissage de la liste avec les mots lematisés
clean_words=nlp(" ".join(clean_words))
for w in clean_words:
   clean_words_lem.append(w.lemma_)




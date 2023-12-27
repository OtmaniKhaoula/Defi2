import csv 
import numpy as np 
import os 
import sys 
import pandas as pd 
import re

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)
print("DIRECTORY", directory)

file = sys.argv[1]

import configg

path = configg.paths['url']
print(path)
comments_train = np.load(f"{path}/processed_data/{file}/comments.npy", allow_pickle=True).item()
#print("structure fichier = ", comments_train)

cpt = 0
dict_emotion = {}
dict = {0: "positif", 1: "neutre", 2:"negatif"}

with open("../polarity_word.txt", encoding = "latin-1") as fichier:
    for ligne in fichier:
        cpt += 1
        #print("cpt = ",  cpt)
        """        
        if(cpt < 220000):
            continue
        elif(cpt > 225000):
            break
        """
        # Séparer la phrase par des point-virgules à l'extérieur des guillemets
        ligne = re.split(';\s*(?=(?:[^"]*"[^"]*")*[^"]*$)', ligne)
        # Retirer les éventuels espaces autour des parties
        ligne_tokenized = [l.strip() for l in ligne]
        ligne_tokenized[1] = ligne_tokenized[1].replace('"', '')
        #print(ligne_tokenized)
        #ligne_tokenized = ligne.split(";")
        if(ligne_tokenized[2] == ligne_tokenized[4]):
            dict_emotion[ligne_tokenized[1]] = "neutre"
            continue
        A = [float(element) for element in ligne_tokenized[2:]]
        idx = np.argmax(A, axis=0)
        #print("word = ", ligne_tokenized[1])
        #print("liste = ", A)
        #print("idx = ", idx)
        #"&#273;&#432;&#7901;i &#432;&#417;i"
        dict_emotion[ligne_tokenized[1]] = dict[idx]
        #print("tokenized = ", ligne_tokenized[1])

print("taille = ", len(dict_emotion))
np.save(f"{path}/processed_data/dict_polarity.npy", dict_emotion)

df = pd.DataFrame(list(dict_emotion.items()))
df.to_csv(f"{path}/processed_data/dict_polarity.csv", encoding = "latin-1", header = False)

#dict_emotion = {"film": "neutre"}
print("film = ", dict_emotion["film"])

# Ajouter les polarités dans les commentaires
cpt = 0
list_keyword = list(dict_emotion.keys())
for review in comments_train.keys():
    review_split = comments_train[review].split(" ")
    for word, i in zip(review_split, [i for i in range(len(review_split))]):
        if (word in list_keyword):
            review_split[i] = word + " " + dict_emotion[word]
            #print("review_split = ", review_split[i])
    cpt += 1
    print("cpt = ", cpt)
    #if cpt % 1000 == 0:
    #    print("cpt = ", cpt)
    #if(cpt>50):
    #    break
    comments_train[review] = " ".join(review_split)
    #print("modified_words = ", comments_train[review])
    """
    for key in dict_emotion.keys():
        #print("key = ", key)
        #print(comments_train[review])
        if(key in comments_train[review]):
            #print(dict_emotion[key])
            words = comments_train[review].split()

            modified_words = []

            for word in words:
                modified_words.append(word)
                if key == word:
                    #print("key = ", key)
                    #print("word = ", word)
                    modified_words.append(dict_emotion[key])

            comments_train[review] = " ".join(modified_words)
            print("modified_words = ", modified_words)
    #print("structure fichier = ", comments_train[review])
    """
np.save(f"{path}/processed_data/{file}/comments_with_polarities.npy", comments_train)

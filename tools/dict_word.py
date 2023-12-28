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

import config

path = config.paths['url']
print(path)

comments = np.load(f"{path}/processed_data/{file}/comments.npy", allow_pickle=True).item()
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
        if(bool(re.search(r"\d", ligne_tokenized[1])) or not bool(re.search('[a-zA-Z]', ligne_tokenized[1])) or len(ligne_tokenized[1].split(" "))>1 or len(ligne_tokenized[1].split(">"))>1):
            continue
        if(ligne_tokenized[2] == ligne_tokenized[4]):
            dict_emotion[ligne_tokenized[1]] = "neutre"
            continue
        A = [float(element) for element in ligne_tokenized[2:]]
        idx = np.argmax(A, axis=0)
        dict_emotion[ligne_tokenized[1]] = dict[idx]

print("taille = ", len(dict_emotion))
np.save(f"{path}/processed_data/dict_polarity.npy", dict_emotion)

df = pd.DataFrame(list(dict_emotion.items()))
df.to_csv(f"{path}/processed_data/dict_polarity.csv", encoding = "latin-1", header = False)

# Ajouter les polarités dans les commentaires
cpt = 0
list_keyword = list(dict_emotion.keys())

for review in comments.keys():
    if isinstance(comments[review], float):
        continue

    review_split = comments[review].split(" ")
    
    modified_review = [word + " " + dict_emotion.get(word, "") if word in list_keyword else word for word in review_split]
    
    comments[review] = " ".join(modified_review)
    print(comments[review])
    
    cpt += 1
    if cpt % 100 == 0:
        print("cpt =", cpt)
    
np.save(f"{path}/processed_data/{file}/comments_with_polarities.npy", comments)

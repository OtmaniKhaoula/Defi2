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
dict_emotion = np.load(f"{path}/processed_data/dict_polarity.npy", allow_pickle=True).item()

# Ajouter les polarités dans les commentaires
cpt = 0
list_keyword = list(dict_emotion.keys())

for review in comments.keys():
    # Vérifier si la valeur est de type float ou None
    if isinstance(comments[review], (float, type(None))):
        continue

    # Vérifier si la valeur est de type str
    if not isinstance(comments[review], str):
        continue

    review_split = comments[review].split(" ")
    
    modified_review = [word + " " + dict_emotion.get(word, "") if word in list_keyword else word for word in review_split]
    
    comments[review] = " ".join(modified_review)
    #print(comments[review])
    
    cpt += 1
    if cpt % 100 == 0:
        print("cpt =", cpt)
    """
    if(cpt>20):
        break    
    """
np.save(f"{path}/processed_data/{file}/comments_with_polarities.npy", comments)

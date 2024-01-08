import csv 
import numpy as np 
import os 
import sys 
import pandas as pd 
import re
from tqdm import tqdm

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)
print("DIRECTORY", directory)

file = sys.argv[1]
print("file", file, flush=True)

import config

path = config.paths['url']
print(path)

comments = np.load(f"{path}/processed_data/{file}/comments.npy", allow_pickle=True).item()
dict_emotion = np.load(f"{path}/processed_data/dict_polarity.npy", allow_pickle=True).item()

# Ajouter les polarités dans les commentaires

for review, comment in comments.items():
    if not isinstance(comment, str) or not comment:
        continue

    modified_review = [word + " " + dict_emotion.get(word, "") if word in dict_emotion else word for word in comment.split(" ")]
    
    comments[review] = " ".join(modified_review)


savfile = f"{path}/processed_data/{file}/comments_with_polarities.npy"
np.save(savfile, comments)

for key in comments:
    print(comments[key], flush=True)
    break


print("savfile", savfile, flush=True)
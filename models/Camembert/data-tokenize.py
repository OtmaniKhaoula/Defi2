import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer, AdamW
import sys, os
import torch.nn as nn

directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(directory)

import config
path = config.paths['url']


if torch.cuda.is_available():
    print("GPU disponible!", flush=True)
else:
    print("Aucun GPU disponible. Vérifiez votre configuration.", flush=True)

# Vérifier la disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("device = ", device)

"""
Chargement du jeu de donnees d'apprentissage
"""
print("Starting data loading…", flush=True)

reviews_grades_train = np.load(f"{path}/processed_data/train/reviews_grades.npy", allow_pickle=True).item()
comments_train = np.load(f"{path}/processed_data/train/comments.npy", allow_pickle=True).item()
reviews_users_train = np.load(f"{path}/processed_data/train/reviews_users.npy", allow_pickle=True).item()
reviews_movie_train = np.load(f"{path}/processed_data/train/reviews_movie.npy", allow_pickle=True).item()

# Convertir le dictionnaire en DataFrame avec une colonne pour les clés et une colonne pour les valeurs
df_comments_train = pd.DataFrame(list(comments_train.items()), columns=['id_reviews', 'comments'])
df_reviews_grades_train = pd.DataFrame(list(reviews_grades_train.items()), columns=['id_reviews', 'notes'])
df_reviews_grades_train['notes'] = df_reviews_grades_train['notes'] * 2 - 1
df_reviews_grades_train['notes'] = df_reviews_grades_train['notes'].astype(int)

# Jointure
data_train = pd.merge(df_comments_train, df_reviews_grades_train, on='id_reviews')
data_train = data_train.iloc[:150, :]

comments_train = data_train['comments'].values.tolist()
notes_train = data_train['notes'].values.tolist()

"""
Chargement du jeu de donnees de développement
"""

reviews_grades_dev = np.load(f"{path}/processed_data/dev/reviews_grades.npy", allow_pickle=True).item()
comments_dev = np.load(f"{path}/processed_data/dev/comments.npy", allow_pickle=True).item()
reviews_users_dev = np.load(f"{path}/processed_data/dev/reviews_users.npy", allow_pickle=True).item()
reviews_movie_dev = np.load(f"{path}/processed_data/dev/reviews_movie.npy", allow_pickle=True).item()

# Convertir le dictionnaire en DataFrame avec une colonne pour les clés et une colonne pour les valeurs
df_comments_dev = pd.DataFrame(list(comments_dev.items()), columns=['id_reviews', 'comments'])
df_reviews_grades_dev = pd.DataFrame(list(reviews_grades_dev.items()), columns=['id_reviews', 'notes'])
df_reviews_grades_dev['notes'] = df_reviews_grades_dev['notes'] * 2 - 1
df_reviews_grades_dev['notes'] = df_reviews_grades_dev['notes'].astype(int)

# Jointure
data_dev = pd.merge(df_comments_dev, df_reviews_grades_dev, on='id_reviews')
data_dev = data_dev.iloc[:30, :]

comments_dev = data_dev['comments'].values.tolist()
notes_dev = data_dev['notes'].values.tolist()

#print("comments_dev = ", comments_dev[:2])
#print("notes_dev = ", notes_dev[:2])

print("End dataloading…", flush=True)

"""
Ajouter les ids des utilisateurs et des films dans les commentaires
"""
"""
data_train_user_movie = data_train.copy()
data_train_user = data_train.copy()
data_train_movie = data_train.copy()
for i in range(data_train_user_movie.shape[0]):
    id_review = data_train_user_movie.loc[i, "id_reviews"]
    id_user = reviews_users_train[id_review]
    id_movie = reviews_movie_train[id_review]
    data_train_user_movie.loc[i, "comments"] = data_train_user_movie.loc[i, "comments"] + id_user + " " + id_movie
    data_train_user.loc[i, "comments"] = data_train_user.loc[i, "comments"] + id_user
    data_train_movie.loc[i, "comments"] = data_train_movie.loc[i, "comments"] + id_movie

data_dev_user_movie = data_dev.copy()
data_dev_user = data_dev.copy()
data_dev_movie = data_dev.copy()
or i in range(data_dev_user_movie.shape[0]):
    id_review = data_dev_user_movie.loc[i, "id_reviews"]
    id_user = reviews_users_dev[id_review]
    id_movie = reviews_movie_dev[id_review]
    data_dev_user_movie.loc[i, "comments"] = data_dev.loc[i, "comments"] + id_user + " " + id_movie
    data_dev_user.loc[i, "comments"] = data_dev_user.loc[i, "comments"] + id_user
    data_dev_movie.loc[i, "comments"] = data_dev_movie.loc[i, "comments"] + id_movie
"""

"""
Encodage du texte
"""
print("Starting tokenization…", flush=True)
# On charge l'objet "tokenizer"de camemBERT qui va servir a encoder
# 'camebert-base' est la version de camembert qu'on choisit d'utiliser
# cmarkea/distilcamembert-base-sentiment
# 'do_lower_case' à True pour qu'on passe tout en miniscule
text = 'cmarkea/distilcamembert-base-sentiment'
#text = 'tblard/tf-allocine'
TOKENIZER = CamembertTokenizer.from_pretrained(
    text,
    do_lower_case=True
    )

"""
import transformers
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
"""

# Taille max d'un commentaire
MAX_LENGTH = 512

comments_train = [comment if comment is not None else "" for comment in comments_train]
comments_dev = [comment if comment is not None else "" for comment in comments_dev]

print("None removed…", flush=True)
#print(type(comments_train))

# La fonction batch_encode_plus encode un batch de donnees
encoded_batch_train = TOKENIZER.batch_encode_plus(comments_train,
                                            add_special_tokens=True,
                                            max_length=MAX_LENGTH,
                                            padding=True,
                                            truncation=True,
                                            return_attention_mask = True,
                                            return_tensors = 'pt')

encoded_batch_dev = TOKENIZER.batch_encode_plus(comments_dev,
                                            add_special_tokens=True,
                                            max_length=MAX_LENGTH,
                                            padding=True,
                                            truncation=True,
                                            return_attention_mask = True,
                                            return_tensors = 'pt')

print("TOKENIZER call end…", flush=True)

print("Converting to tensor…", flush=True)

# On transforme la liste des sentiments en tenseur
notes_train = torch.tensor(notes_train).to(device)
notes_dev = torch.tensor(notes_dev).to(device)
 
# On met nos données sous forme de TensorDataset
train_dataset = TensorDataset(
    encoded_batch_train['input_ids'],
    encoded_batch_train['attention_mask'],
    notes_train)
dev_dataset = TensorDataset(
    encoded_batch_dev['input_ids'],
    encoded_batch_dev['attention_mask'],
    notes_dev)

torch.save(train_dataset, f"{path}/models/BERT/camembert_tokenized_data/train_dataset")
torch.save(dev_dataset, f"{path}/models/BERT/camembert_tokenized_data/dev_dataset")

print("End tokenisation…", flush=True)
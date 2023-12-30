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
data_train = data_train.iloc[:, :]

comments_train = data_train['comments'].values.tolist()
notes_train = data_train['notes'].values.tolist()

#Chargement du jeu de donnees de développement


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
data_dev = data_dev.iloc[:, :]

comments_dev = data_dev['comments'].values.tolist()
notes_dev = data_dev['notes'].values.tolist()

#### TEST ####

comments_test = np.load(f"{path}/processed_data/test/comments_clean.npy", allow_pickle=True).item()
metadata_test = np.load(f"{path}/processed_data/test/movie_metadata.npy", allow_pickle=True).item()
review_movie_test = np.load(f"{path}/processed_data/test/reviews_movie.npy", allow_pickle=True).item()
reviews_users_test = np.load(f"{path}/processed_data/test/reviews_users.npy", allow_pickle=True).item()

notes_test = []
key_dict = {}
k = {}

for key in comments_test:
    if key not in k:
        key_dict[len(k)] = key
        k[key] = len(k)

        notes_test.append(k[key])

np.save(f"{path}/models/Camembert/camembert_tokenized_data/k.npy", k)
np.save(f"{path}/models/Camembert/camembert_tokenized_data/key_dict.npy", key_dict)

comments_test_list = []
for key in comments_test:
    movie = review_movie_test[key]
    
    if not comments_test[key] == "" and not reviews_users_test[key] == "":
        comments_test_list.append(comments_test[key] + [reviews_users_test[key]] + metadata_test[movie].split(" ") + [movie])
    else:
        comments_test_list.append([comments_test[key]] + [reviews_users_test[key]] + metadata_test[movie].split(" ") + [movie])

print("End dataloading…", flush=True)

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


# Taille max d'un commentaire
MAX_LENGTH = 512

#comments_train = [comment if comment is not None else "" for comment in comments_train]
#comments_dev = [comment if comment is not None else "" for comment in comments_dev]

print(comments_test_list[0], flush=True)
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

encoded_batch_test = TOKENIZER.batch_encode_plus(comments_test,
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
notes_test = torch.tensor(notes_test).to(device)
 
# On met nos données sous forme de TensorDataset
train_dataset = TensorDataset(
    encoded_batch_train['input_ids'],
    encoded_batch_train['attention_mask'],
    notes_train)

dev_dataset = TensorDataset(
    encoded_batch_dev['input_ids'],
    encoded_batch_dev['attention_mask'],
    notes_dev)

test_dataset = TensorDataset(
    encoded_batch_test['input_ids'],
    encoded_batch_test['attention_mask'],
    notes_test)

torch.save(train_dataset, f"{path}/models/Camembert/camembert_tokenized_data/train_dataset")
torch.save(dev_dataset, f"{path}/models/Camembert/camembert_tokenized_data/dev_dataset")
torch.save(test_dataset, f"{path}/models/Camembert/camembert_tokenized_data/test_dataset")

print("End tokenisation…", flush=True)
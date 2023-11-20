# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:01:25 2023

@author: Elisa Martin

Lien: https://ledatascientist.com/analyse-de-sentiments-avec-camembert/
"""

"""
Importation de librairies
"""

#import torch
import seaborn
import pandas as pd
import numpy as np
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer, AdamW
import sys, os
import config

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)

path = config.paths['url']

"""
Chargement du jeu de donnees d'apprentissage
"""

reviews_grades_train = np.load(f"{path}/processed_data/train/reviews_grades.npy", allow_pickle=True).item()
comments_train = np.load(f"{path}/processed_data/train/comments.npy", allow_pickle=True).item()

# Convertir le dictionnaire en DataFrame avec une colonne pour les clés et une colonne pour les valeurs
df_comments_train = pd.DataFrame.from_dict(comments_train, orient='index', columns=['id_reviews', 'comments']).reset_index()
df_reviews_grades_train = pd.DataFrame.from_dict(reviews_grades_train, orient='index', columns=['id_reviews', 'notes']).reset_index()

# Jointure 
data_train = pd.merge(df_comments_train, df_reviews_grades_train, on='id_reviews')

comments_train = data_train['comments'].values.tolist()
notes_train = data_train['notes'].values.tolist()

"""
Chargement du jeu de donnees de développement
"""

reviews_grades_dev = np.load(f"{path}/processed_data/dev/reviews_grades.npy", allow_pickle=True).item()
comments_dev = np.load(f"{path}/processed_data/dev/comments.npy", allow_pickle=True).item()

# Convertir le dictionnaire en DataFrame avec une colonne pour les clés et une colonne pour les valeurs
df_comments_dev = pd.DataFrame.from_dict(comments_dev, orient='index', columns=['id_reviews', 'comments']).reset_index()
df_reviews_grades_dev = pd.DataFrame.from_dict(reviews_grades_dev, orient='index', columns=['id_reviews', 'notes']).reset_index()

# Jointure 
data_dev = pd.merge(df_comments_dev, df_reviews_grades_dev, on='id_reviews')

comments_dev = data_dev['comments'].values.tolist()
notes_dev = data_dev['notes'].values.tolist()


"""
Encodage du texte
"""

# On charge l'objet "tokenizer"de camemBERT qui va servir a encoder
# 'camebert-base' est la version de camembert qu'on choisit d'utiliser
# 'do_lower_case' à True pour qu'on passe tout en miniscule
TOKENIZER = CamembertTokenizer.from_pretrained(
    'camembert-base',
    do_lower_case=True)

# Taille max d'un commentaire
MAX_LENGTH = 512
 
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


# On transforme la liste des sentiments en tenseur
notes_train = torch.tensor(notes_train) 
notes_dev = torch.tensor(notes_dev)
 
# On met nos données sous forme de TensorDataset
train_dataset = TensorDataset(
    encoded_batch_train['input_ids'],
    encoded_batch_train['attention_mask'],
    notes_train)
validation_dataset = TensorDataset(
    encoded_batch_dev['input_ids'],
    encoded_batch_dev['attention_mask'],
    notes_dev)
 
batch_size = 32
 
# On cree les DataLoaders d'entrainement et de validation
# Le dataloader est juste un objet iterable
# On le configure pour iterer le jeu d'entrainement de façon aleatoire et creer les batchs.
train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size)
 
validation_dataloader = DataLoader(
            validation_dataset,
            sampler = SequentialSampler(validation_dataset),
            batch_size = batch_size)

"""
Chargement du modèle
"""
# On la version pre-entrainee de camemBERT 'base'
model = CamembertForSequenceClassification.from_pretrained(
    'camembert-base',
    num_labels = 10)

"""
Hyperparamètres
"""
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # Learning Rate
                  eps = 1e-8, # Epsilon)
                  epochs = 3)

epochs = 3

"""
Entraînement
"""
# On va stocker nos tensors sur mon cpu : je n'ai pas mieux
device = torch.device("cpu")
 
# Pour enregistrer les stats a chaque epoque
training_stats = []
 
# Boucle d'entrainement
for epoch in range(0, epochs):
     
    print("")
    print(f'########## Epoch {epoch+1} / {epochs} ##########')
    print('Training...')
 
 
    # On initialise la loss pour cette epoque
    total_train_loss = 0
 
    # On met le modele en mode 'training'
    # Dans ce mode certaines couches du modele agissent differement
    model.train()
 
    # Pour chaque batch
    for step, batch in enumerate(train_dataloader):
 
        # On fait un print chaque 40 batchs
        if step % 40 == 0 and not step == 0:
            print(f'  Batch {step}  of {len(train_dataloader)}.')
         
        # On recupere les donnees du batch
        input_id = batch[0].to(device)
        attention_mask = batch[1].to(device)
        sentiment = batch[2].to(device)
 
        # On met le gradient a 0
        model.zero_grad()        
 
        # On passe la donnee au model et on recupere la loss et le logits (sortie avant fonction d'activation)
        loss, logits = model(input_id, 
                             token_type_ids=None, 
                             attention_mask=attention_mask, 
                             labels=sentiment)
 
        # On incremente la loss totale
        # .item() donne la valeur numerique de la loss
        total_train_loss += loss.item()
 
        # Backpropagtion
        loss.backward()
 
        # On actualise les parametrer grace a l'optimizer
        optimizer.step()
 
    # On calcule la  loss moyenne sur toute l'epoque
    avg_train_loss = total_train_loss / len(train_dataloader)   
 
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))  
     
    # Enregistrement des stats de l'epoque
    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
        }
    )
 
print("Model saved!")
torch.save(model.state_dict(), "./sentiments.pt")





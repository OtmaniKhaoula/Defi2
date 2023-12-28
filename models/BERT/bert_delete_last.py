# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:01:25 2023

@author: Elisa Martin & Khaoula Otmani

Lien: https://ledatascientist.com/analyse-de-sentiments-avec-camembert/
"""
"""
Importation de librairies
"""

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
    print("GPU disponible!")
else:
    print("Aucun GPU disponible. Vérifiez votre configuration.")

# Vérifier la disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("device = ", device)

"""
Chargement du jeu de donnees d'apprentissage
"""

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
data_train = data_train.iloc[:1500, :]

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
data_dev = data_dev.iloc[:300, :]

comments_dev = data_dev['comments'].values.tolist()
notes_dev = data_dev['notes'].values.tolist()

#print("comments_dev = ", comments_dev[:2])
#print("notes_dev = ", notes_dev[:2])

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

for i in range(len(comments_train)):
    if i % 10000 == 0:
        print(f"Iteration {i}")
    if comments_train[i] is None:
        comments_train[i] = ""

for i in range(len(comments_dev)):
    if i % 10000 == 0:
        print(f"Iteration {i}")
    if comments_dev[i] is None:
        comments_dev[i] = ""

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


# On transforme la liste des sentiments en tenseur
notes_train = torch.tensor(notes_train).to(device)
notes_dev = torch.tensor(notes_dev).to(device)
 
# On met nos données sous forme de TensorDataset
train_dataset = TensorDataset(
    encoded_batch_train['input_ids'],
    encoded_batch_train['attention_mask'],
    notes_train)
validation_dataset = TensorDataset(
    encoded_batch_dev['input_ids'],
    encoded_batch_dev['attention_mask'],
    notes_dev)
 
batch_size = 8
 
# On cree les DataLoaders d'entrainement et de validation
# Le dataloader est juste un objet iterable
# On le configure pour iterer le jeu d'entrainement de façon aleatoire et creer les batchs.
train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size)

val_dataloader = DataLoader(
            validation_dataset,
            sampler = SequentialSampler(validation_dataset),
            batch_size = batch_size)

"""
Chargement du modèle
"""
# On prend la version pre-entrainee de DistilcamemBERT sur les sentiments
model = CamembertForSequenceClassification.from_pretrained(text).to(device)
print("model = ", model)
# Ajouter une nouvelle couche pour 10 classes (remplacement de la dernière couche
#new_classifier_layer = nn.Linear(model.config.hidden_size, 10).to(device)

# Remplacer la dernière couche du modèle par la nouvelle couche de classification
#model.classifier = new_classifier_layer
# Changer le nombre de classes dans la dernière couche du classificateur
model.classifier.out_proj = nn.Linear(in_features=768, out_features=10, bias=True).to(device)
#model.classifier = nn.Linear(model.config.dim, 9)

# Initialiser les poids de la nouvelle couche
#nn.init.xavier_uniform_(model.classifier.out_proj.weight).to(device)

# Afficher la structure du modèle
print(model)
#print(model.classifier.weight.size())
"""
Hyperparamètres
"""
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epochs = 10
early_stopping_patience = 2
epochs_without_improvement = 0
best_val_loss = 0

loss = nn.CrossEntropyLoss().to(device)

for param in model.parameters():
    param.requires_grad_(True)
"""
Entraînement
"""

training_stats = []

# For each epoch...
for epoch in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Afficher la taille des tenseurs dans chaque batch
        print(f"Batch {step + 1} - Taille des tenseurs:", flush=True)
        #print(f"  Input ID: {b_input_ids.size()}")
        #print(f"  Attention Mask: {b_input_mask.size()}")
        #print(f"  Sentiment: {b_labels.size()}")
        model.zero_grad()
        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        #loss = F.cross_entropy(output.logits, b_labels)
        #print( "output.logits = ", output.logits)

        #print("logit corrige = ", output.logits.view(-1, output.logits.size(), "target corrigé = ", b_labels.view(-1).size())

        #print("output = ", output)
        #print(output.logits.dtype)
        # Appliquer la fonction softmax sur la dernière dimension (axis=-1)
        probs = nn.functional.softmax(output.logits, dim=-1)
        #print("probs = ", probs) 
        # Trouver la classe prédite pour chaque exemple dans le batch
        predicted_classes = torch.argmax(probs, dim=-1)
        
        print("predicted_classe = ",predicted_classes.long(), flush=True)
        print(" b_labels = ", b_labels.long(), flush=True)
        loss = nn.CrossEntropyLoss()(predicted_classes.float(), b_labels.float())        
        #print("loss = ", loss)
        #output = model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask,labels=b_labels)
        #loss = output.loss
        total_train_loss += loss.item()
        # Perform a backward pass to calculate the gradients
        loss = loss.requires_grad_()
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)               
    print("", flush=True)
    print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
    
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("", flush=True)
    print("Running Validation...", flush=True)
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    # Tracking variables 
    total_eval_loss = 0
    nb_eval_steps = 0
    # Evaluate data for one epoch
    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        
            output = model(b_input_ids,
                           token_type_ids=None, 
                           attention_mask=b_input_mask)
        
        probs = nn.functional.softmax(output.logits, dim=-1)
        #print("probs = ", probs)
        # Trouver la classe prédite pour chaque exemple dans le batch
        predicted_classes = torch.argmax(probs, dim=-1)

        print("predicted_classe = ",predicted_classes.long(), flush=True)
        print(" b_labels = ", b_labels.long(), flush=True)
        loss = nn.CrossEntropyLoss()(predicted_classes.float(), b_labels.float())
        total_eval_loss += loss.item()
        # Move logits and labels to CPU if we are using GPU
        #logits = output.logits
        #logits = logits.detach().cpu().numpy()
        #label_ids = b_labels.to('cpu').numpy()
        
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(val_dataloader)
    """
    if avg_val_accuracy > best_eval_accuracy:
        torch.save(model, 'bert_model')
        best_eval_accuracy = avg_val_accuracy
    """
    print("  Validation Loss: {0:.2f}".format(avg_val_loss), flush=True)
    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
        }
    )
     # Vérification de l'amélioration
    if epoch == 0 or avg_val_loss < best_val_loss:
        # Si la perte de validation s'améliore, enregistrez le modèle et réinitialisez le compteur
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "./sentiments_test.pt")
    else:
        # Sinon, incrémentez le compteur
        epochs_without_improvement += 1

    # Vérification de l'arrêt anticipé
    if epochs_without_improvement >= early_stopping_patience:
        print(f"Arrêt anticipé après {epoch+1} époques sans amélioration.", flush=True)
        break

print("ts = ", training_stats, flush=True)

print("Model saved!", flush=True)
torch.save(model.state_dict(), "./sentiments_test.pt")



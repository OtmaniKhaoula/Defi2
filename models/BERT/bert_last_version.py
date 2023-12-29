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

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)

#path = config.paths['url']
#path = "C:/Users/Utilisateur/Documents/M2/Application_innovation/Sujet_2/new_Defi2"

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
data_train = data_train.iloc[:40, :]

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
data_dev = data_dev.iloc[:24, :]

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

"""
import transformers
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
"""

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
 
batch_size = 32
 
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
# distilBERT tokenizer
#config = transformers.DistilBertConfig(dropout=0.2, attention_dropout=0.2)
#dbert_pt = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

# On prend la version pre-entrainee de DistilcamemBERT sur les sentiments
model = CamembertForSequenceClassification.from_pretrained(text).to(device)
# Appliquez la fonction ReLU
model.classifier.out_proj = nn.Linear(in_features=768, out_features=10, bias=True)
# Ajoutez la couche ReLU entre les couches linéaires
new_structure = nn.Sequential(
    nn.Dropout(p=0.2, inplace=False),
    nn.Linear(in_features=768, out_features=768, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=768, out_features=10, bias=True)
)
model.classifier = new_structure

# Let's create a sample of size 5 from the training data
#sample = train_dataset[0:5]

"""
print('Object type: ', type(dbert_pt(sample)))
print('Output format (shape): ',dbert_pt(sample)[0].shape)
print('Output used as input for the classifier (shape): ', dbert_pt(sample)[0][:,0,:].shape)
"""
#print('Object type: ', type(model(sample[0], sample[1])))
# Object type:  <class 'transformers.modeling_outputs.SequenceClassifierOutput'>
#print('Logit Output format (shape): ',model(sample[0], sample[1]).logits.shape)
# Logit Output format (shape):  torch.Size([5, 10])
#print("Logit = ", model(sample[0], sample[1]).logits)
# 5 vecteurs de cette forme:
# [-0.5281,  0.1070,  0.1522,  0.1715,  0.6342,  0.0482,  0.1057, -0.2173, 0.5730,  0.5887]
# [5, 5, 5, 5, 5]
#print("sample sortie = ", sample[2])
# sample sortie =  tensor([7, 6, 8, 5, 6])

# Dans le cas où on a ajouté relu et qu'on veut la sortie pour chaque exemple
#print('Output used as input for the classifier (shape): ', model(sample[0], sample[1])[0][:,0,:].shape)

for param in model.parameters():
    param.requires_grad = False
    
# Regardons le nombre de paramètres (entraînables et non entraînables) :
#total_params = sum(p.numel() for p in model.parameters())
#total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print("Number of parameters: ", total_params)
#print("Number of trainable parameters: ", total_params_trainable)

for param in model.classifier.parameters():
    param.requires_grad = True
  
# Regardons le nombre de paramètres (entraînables et non entraînables) :
#total_params = sum(p.numel() for p in model.parameters())
#total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print("Number of parameters: ", total_params)
#print("Number of trainable parameters: ", total_params_trainable)

epochs = 15
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Define the dictionary "history" that will collect key performance indicators during training
epoch=[]
train_loss=[]
valid_loss=[]
train_accuracy=[]
valid_accuracy=[]

best_val_loss = -1000000000000
early_stopping_patience = 3
epochs_without_improvement = 0

# Measure time for training
from datetime import datetime
start_time = datetime.now()

# Loop on epochs
for e in range(epochs):
    
    # Set mode in train mode
    model.train()
    
    train_loss_sum = 0.0
    train_accuracy = []
    
    # Loop on batches
    for step, batch in enumerate(train_dataloader):
        X = batch[0].to(device)
        input_mask = batch[1].to(device)
        y = batch[2].to(device)
        
        # Get prediction & loss
        #print("Shape de X = ", X.shape)
        prediction = model(X, input_mask)
        #print("prediction = ", prediction[0].shape)
        prediction = prediction[0][:,0,:]
        #print("output = ", prediction.shape)
        prediction_probs = nn.functional.softmax(prediction, dim=-1)
        predicted_classes = torch.argmax(prediction_probs, dim=-1)
        #print("prediction = ", prediction)
        #print("Y = ", y, "predicted_classes = ", predicted_classes)
        #loss1 = criterion(predicted_classes.float(), y.float())
        #print("loss1 = ", loss1)
        loss = criterion(prediction_probs, y)
        #print("loss = ", loss)
        # Adjust the parameters of the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss_sum += loss.item()
        
        prediction_index = prediction.argmax(axis=1)
        accuracy = (prediction_index==y)
        train_accuracy += accuracy
    
    train_accuracy_value = (sum(train_accuracy) / len(train_accuracy)).item()
    
    # Calculate the loss on the test data after each epoch
    # Set mode to evaluation (by opposition to training)
    model.eval()
    valid_loss_sum = 0.0
    valid_accuracy = []
    for step, batch in enumerate(val_dataloader):
        X = batch[0].to(device)
        input_mask = batch[1].to(device)
        y = batch[2].to(device)
         
        prediction = model(X, input_mask)
        prediction = prediction[0][:,0,:]
        #print("output = ", prediction.shape)
        prediction_probs = nn.functional.softmax(prediction, dim=-1)
        predicted_classes = torch.argmax(prediction_probs, dim=-1)
        #print("prediction2 = ", prediction)
        print("Y2 = ", y, "predicted_classes = ", predicted_classes)
        #loss1 = criterion(predicted_classes.float(), y.float())
        #print("loss2 = ", loss1)
        loss = criterion(prediction_probs, y)
        #print("loss = ", loss)

        valid_loss_sum += loss.item()
        
        prediction_index = prediction.argmax(axis=1)
        accuracy = (prediction_index==y)
        valid_accuracy += accuracy
    valid_accuracy_value = (sum(valid_accuracy) / len(valid_accuracy)).item()
    
    # Populate history
    epoch.append(e+1)
    train_loss.append(train_loss_sum / len(train_dataloader))
    valid_loss.append(valid_loss_sum / len(val_dataloader))
    train_accuracy.append(train_accuracy_value)
    valid_accuracy.append(valid_accuracy_value)    
        
    print(f'Epoch {e+1} \t\t Training Loss: {train_loss_sum / len(train_dataloader) :10.3f} \t\t Validation Loss: {valid_loss_sum / len(val_dataloader) :10.3f}')
    print(f'\t\t Training Accuracy: {train_accuracy_value :10.3%} \t\t Validation Accuracy: {valid_accuracy_value :10.3%}')
    
    # Vérification de l'amélioration
    #print("best = ", best_val_loss)
    #print("valid_accuracy_value  = ", valid_loss_sum)
    if e == 0 or valid_loss_sum < best_val_loss:
        # Si la perte de validation s'améliore, enregistrez le modèle et réinitialisez le compteur
        best_val_loss = valid_loss_sum
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "./sentiments.pt")
    else:
        # Sinon, incrémentez le compteur
        epochs_without_improvement += 1

    # Vérification de l'arrêt anticipé
    if epochs_without_improvement >= early_stopping_patience:
        print(f"Arrêt anticipé après {e+1} époques sans amélioration.")
        break

# Measure time for training
end_time = datetime.now()
training_time_pt = (end_time - start_time).total_seconds()    










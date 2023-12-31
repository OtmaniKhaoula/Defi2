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
    print("GPU disponible!", flush=True)
else:
    print("Aucun GPU disponible. Vérifiez votre configuration.", flush=True)

# Vérifier la disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("device = ", device)

batch_size = 32
text = 'cmarkea/distilcamembert-base-sentiment'

# On cree les DataLoaders d'entrainement et de validation
# Le dataloader est juste un objet iterable
# On le configure pour iterer le jeu d'entrainement de façon aleatoire et creer les batchs.
print("Loading tokenized dataset…", flush=True)
train_dataset = torch.load(f"{path}/models/BERT/camembert_tokenized_data/train_dataset")
validation_dataset = torch.load(f"{path}/models/BERT/camembert_tokenized_data/dev_dataset")

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
print("Loading model…", flush=True)
# distilBERT tokenizer
#config = transformers.DistilBertConfig(dropout=0.2, attention_dropout=0.2)
#dbert_pt = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

# On prend la version pre-entrainee de DistilcamemBERT sur les sentiments
model = CamembertForSequenceClassification.from_pretrained(text).to(device)
# Appliquez la fonction ReLU
model.classifier.out_proj = nn.Linear(in_features=768, out_features=10, bias=True).to(device)
# Ajoutez la couche ReLU entre les couches linéaires
new_structure = nn.Sequential(
    nn.Dropout(p=0.2, inplace=False),
    nn.Linear(in_features=768, out_features=768, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=768, out_features=10, bias=True)
).to(device)

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, eps=1e-8)

# Define the dictionary "history" that will collect key performance indicators during training
epoch=[]
train_loss=[]
valid_loss=[]
train_accuracy_complete=[]
valid_accuracy_complete=[]

best_val_loss = -1000000000000
early_stopping_patience = 5
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
        #prediction_probs = nn.functional.softmax(prediction, dim=-1)
        #predicted_classes = torch.argmax(prediction_probs, dim=-1)
        loss = criterion(prediction, y)
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
    with torch.no_grad():
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
            #print("Y2 = ", y, "predicted_classes = ", predicted_classes)
            loss = criterion(prediction, y)
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
    train_accuracy_complete.append(train_accuracy_value)
    valid_accuracy_complete.append(valid_accuracy_value)    
        
    print(f'Epoch {e+1} \t\t Training Loss: {train_loss_sum / len(train_dataloader) :10.3f} \t\t Validation Loss: {valid_loss_sum / len(val_dataloader) :10.3f}', flush = True)
    print(f'\t\t Training Accuracy: {train_accuracy_value :10.3%} \t\t Validation Accuracy: {valid_accuracy_value :10.3%}', flush = True)
    
    # Vérification de l'amélioration
    #print("best = ", best_val_loss)
    #print("valid_accuracy_value  = ", valid_loss_sum)
    if e == 0 or (valid_loss_sum / len(val_dataloader)) < best_val_loss:
        # Si la perte de validation s'améliore, enregistrez le modèle et réinitialisez le compteur
        best_val_loss = np.mean(valid_loss_sum / len(val_dataloader))
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "./sentiments.pt")
    else:
        # Sinon, incrémentez le compteur
        epochs_without_improvement += 1

    # Vérification de l'arrêt anticipé
    if epochs_without_improvement >= early_stopping_patience:
        print(f"Arrêt anticipé après {e+1} époques sans amélioration.", flush = True)
        break

# Measure time for training
end_time = datetime.now()
training_time_pt = (end_time - start_time).total_seconds()
print(training_time_pt, flush = True)




















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
test_dataset = torch.load(f"{path}/models/Camembert/camembert_tokenized_data/test_dataset")

test_dataloader = DataLoader(
            test_dataset,
            sampler = SequentialSampler(test_dataset),
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
model.classifier.out_proj = nn.Linear(in_features=768, out_features=10, bias=True)
# Ajoutez la couche ReLU entre les couches linéaires
new_structure = nn.Sequential(
    nn.Dropout(p=0.2, inplace=False),
    nn.Linear(in_features=768, out_features=768, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=768, out_features=10, bias=True)
)
model.classifier = new_structure
# Charger le dictionnaire d'état dans le modèle
model.load_state_dict(torch.load("sentiments-batch256-150000train.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# Measure time for training
from datetime import datetime
start_time = datetime.now()

print("Training…", flush=True)
# Loop on epochs

preds = []
labels = []

for step, batch in enumerate(test_dataloader):
    model.eval()

    X = batch[0].to(device)
    input_mask = batch[1].to(device)
    y = batch[2].to(device)
         
    prediction = model(X, input_mask)
    prediction = prediction[0][:,0,:]
    prediction_probs = nn.functional.softmax(prediction, dim=-1)
    predicted_classes = torch.argmax(prediction_probs, dim=-1)
    preds.append(predicted_classes.tolist())
    labels.append(y.tolist())

# Measure time for training
end_time = datetime.now()
training_time_pt = (end_time - start_time).total_seconds()  

key_dict = np.load(f"{path}/models/Camembert/camembert_tokenized_data/key_dict.npy", allow_pickle=True).item()

to_save = ""

for i in range(len(preds)):
    for j in range(len(preds[i])):
        grade = preds[i][j] + 1
        to_save += f"{key_dict[labels[i][j]]} {str(float(grade)/float(2)).replace('.', ',')}\n"
        
target = open("../../predictions/Camembert-predictions.txt", "w")
target.write(to_save)
target.close()

print("Finished.", flush=True)










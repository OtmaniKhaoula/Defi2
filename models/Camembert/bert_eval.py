"""
Source du site internet: https://ledatascientist.com/analyse-de-sentiments-avec-camembert/
"""
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer, AdamW
import sys, os
import torch.nn as nn
from transformers import DistilBertForSequenceClassification
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import random

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

comments_test = np.load(f"{path}/processed_data/test/comments.npy", allow_pickle=True).item()

"""selected_ids = random.sample(comments_test.keys(), 40)
# Sélectionnez les commentaires correspondant aux identifiants choisis
comments_test = {id: comments_test[id] for id in selected_ids}"""


text = 'cmarkea/distilcamembert-base-sentiment'
#text = 'tblard/tf-allocine'
TOKENIZER = CamembertTokenizer.from_pretrained(
    text,
    do_lower_case=True
    )

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

model.classifier = new_structure.to(device)
# Charger le dictionnaire d'état dans le modèle
model.load_state_dict(torch.load("sentiments-batch256-150000train.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model = model.to(device)
#notes_dev = data_dev['notes'].values.tolist()
def preprocess(raw_reviews, ids=None, sentiments=None):
    encoded_batch = TOKENIZER.batch_encode_plus(raw_reviews,
                                                truncation=True,
                                                pad_to_max_length=True,
                                                return_attention_mask=True,
                                                return_tensors='pt')
    if sentiments:
        sentiments = torch.tensor(sentiments)
        return encoded_batch['input_ids'], encoded_batch['attention_mask'], sentiments, ids
    return encoded_batch['input_ids'], encoded_batch['attention_mask'], ids


def predict(reviews_and_ids, model=model):
    to_save = ""

    with torch.no_grad():
        model.eval()
        reviews_and_ids_2 = list(reviews_and_ids.items())
        #print(reviews_and_ids_2)
        ids, reviews = zip(*reviews_and_ids_2)
        all_input_ids, all_attention_mask, all_ids = preprocess(reviews, ids=ids)

        start = 0
        incr = 256

        while start < len(all_input_ids):
            input_ids = all_input_ids[start:start+incr].to(device)
            attention_mask = all_attention_mask[start:start+incr].to(device)
            ids = all_ids[start:start+incr]

            start += incr

            prediction = model(input_ids, attention_mask=attention_mask)
            prediction = prediction[0][:,0,:]

            predicted_classes = torch.argmax(prediction, dim=-1)
            #print(predicted_classes)
            predictions_with_ids = list(zip(ids, predicted_classes.tolist()))

            for el in predictions_with_ids:
                to_save += f"{el[0]} {str(float(int(el[1])+1)/float(2)).replace('.', ',')}\n"

            print(to_save, flush=True)
            del input_ids, attention_mask, ids

        return predictions_with_ids, to_save

def evaluate(reviews, sentiments):
    predictions = predict(reviews)
    print(f1_score(sentiments, predictions, average='weighted', zero_division=0))
    sns.heatmap(confusion_matrix(sentiments, predictions))

predictions_with_ids, to_save = predict(comments_test, model)

target = open("../../predictions/Camembert-predictions.txt", "w")
target.write(to_save)
target.close()

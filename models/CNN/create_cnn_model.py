import os
import sys
import torch
import fasttext
import numpy as np
import pandas as pd
import nltk

import torch.optim as optim

import time
from nltk.tokenize import word_tokenize

from CNN_NLP import CNN_NLP
from utils import train, data_loader, tokenize_and_encode

#nltk.download("all")

directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(directory)

import config
path = config.paths['url']

#CODE BASED ON THE FOLLOWING TUTORIAL: https://www.kaggle.com/code/williamlwcliu/cnn-text-classification-pytorch/notebook


if torch.cuda.is_available():
    print("GPU disponible!")
else:
    print("Aucun GPU disponible. Vérifiez votre configuration.")

# Vérifier la disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################   IMPORT OUR PRE PROCESSED DATA   ###############################

start = time.time()

reviews_grades_train = np.load(f"{path}/processed_data/train/reviews_grades.npy", allow_pickle=True).item()
comments_train = np.load(f"{path}/processed_data/train/comments_clean.npy", allow_pickle=True).item()
metadata_train = np.load(f"{path}/processed_data/train/movie_metadata.npy", allow_pickle=True).item()
review_movie_train = np.load(f"{path}/processed_data/train/reviews_movie.npy", allow_pickle=True).item()
reviews_users_train = np.load(f"{path}/processed_data/train/reviews_users.npy", allow_pickle=True).item()

reviews_grades_dev = np.load(f"{path}/processed_data/dev/reviews_grades.npy", allow_pickle=True).item()
comments_dev = np.load(f"{path}/processed_data/dev/comments_clean.npy", allow_pickle=True).item()
metadata_dev = np.load(f"{path}/processed_data/dev/movie_metadata.npy", allow_pickle=True).item()
review_movie_dev = np.load(f"{path}/processed_data/dev/reviews_movie.npy", allow_pickle=True).item()
reviews_users_dev = np.load(f"{path}/processed_data/dev/reviews_users.npy", allow_pickle=True).item()


i = 0
for key in comments_train:

    if not comments_train[key] == "" and not reviews_users_train[key] == "":
        comments_train[key] = comments_train[key] + [reviews_users_train[key]] + metadata_train[movie].split(" ") + [movie]


for key in comments_dev:
    movie = review_movie_dev[key]
    
    if not comments_dev[key] == "" and not reviews_users_dev[key] == "":
        comments_dev[key] = comments_dev[key] + [reviews_users_dev[key]] + metadata_dev[movie].split(" ") + [movie]


print("DATA LOADING:", time.time()-start, flush=True)

def initialize_model(pretrained_embedding=None, freeze_embedding=False, vocab_size=None,
                    embed_dim=300, filter_sizes=[3, 4, 5], num_filters=[100, 100, 100],
                    num_classes=10, dropout=0.5, learning_rate=0.01, weight_decay=0):

    assert (len(filter_sizes) == len(num_filters)), "filter_sizes and \
    num_filters need to be of the same length."

    # Instantiate CNN model
    cnn_model = CNN_NLP(pretrained_embedding=pretrained_embedding,
                        freeze_embedding=freeze_embedding,
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        num_classes=10,
                        dropout=0.5)
    
    # Send model to `device` (GPU/CPU)
    cnn_model.to(device)

    # Instantiate Adadelta optimizer
    optimizer = optim.Adadelta(cnn_model.parameters(),
                               lr=learning_rate,
                               rho=0.95,
                               weight_decay=weight_decay)

    return cnn_model, optimizer

def main():
    # Create a directory named 'models' if it doesn't exist
    os.makedirs('./models', exist_ok=True)

    # need to create dir: "./models"
    # CNN-rand: Word vectors are randomly initialized.
    # set_seed(42)

    start = time.time()
    max_len = 2677
    word2idx = {}
    train_dataloader, vocab_size, word2idx = tokenize_and_encode(comments_train, reviews_grades_train, max_len, word2idx)
    dev_dataloader, vocab_size, word2idx = tokenize_and_encode(comments_dev, reviews_grades_dev, max_len, word2idx)
    print("VOCAB SIZE", vocab_size, flush=True)

    """test_dataloader, vocab_size, word2idx = tokenize_and_encode(comments_test, {}, max_len, word2idx, test=True)
    print("FINAL VOCAB SIZE", vocab_size, flush=True)"""

    print("GETTING dataloaders:", time.time()-start, flush=True)

    cnn_rand, optimizer = initialize_model(vocab_size=vocab_size,
                                        embed_dim=300,
                                        learning_rate=1,
                                        dropout=0.5,
                                        weight_decay=1e-3)

    cnn_rand = cnn_rand.to(device)


    print("STARTING TRAINING…", flush=True)
    start = time.time()
    best_acc, train_time = train(cnn_rand, optimizer, train_dataloader, dev_dataloader, epochs=100, model_name="lr=1-emb-dim=300-all-meta")
    print("END OF TRAINING:", time.time()-start, flush=True)

main()
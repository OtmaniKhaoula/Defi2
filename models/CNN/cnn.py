"""
CNN avec Pytorch pour classification des sentiments
Lien du code: https://www.kaggle.com/code/williamlwcliu/cnn-text-classification-pytorch/notebook
Otmani Khaoula & Martin Elisa
"""

import os
import sys
import torch
import fasttext
import tqdm
import numpy as np
import pandas as pd
import nltk

import torch.optim as optim

import time
from nltk.tokenize import word_tokenize

from CNN_NLP import CNN_NLP
from utils import train, evaluate, tokenize, encode, load_pretrained_vectors, data_loader

#nltk.download("all")


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

###############################   IMPORT OUR PRE PROCESSED DATA   ###############################

start = time.time()
reviews_grades_train = np.load(f"{path}/processed_data/train/reviews_grades.npy", allow_pickle=True).item()
comments_train = np.load(f"{path}/processed_data/train/comments.npy", allow_pickle=True).item()
reviews_users_train = np.load(f"{path}/processed_data/train/reviews_users.npy", allow_pickle=True).item()
reviews_movie_train = np.load(f"{path}/processed_data/train/reviews_movie.npy", allow_pickle=True).item()


"""
comments_test = np.load(f"{path}/processed_data/test/comments.npy", allow_pickle=True).item()
data_test = pd.DataFrame(list(comments_test.items()), columns=['id_reviews', 'comments'])
comments_test = data_test['comments'].values.tolist()
"""

reviews_grades_dev = np.load(f"{path}/processed_data/dev/reviews_grades.npy", allow_pickle=True).item()
comments_dev = np.load(f"{path}/processed_data/dev/comments.npy", allow_pickle=True).item()
reviews_users_dev = np.load(f"{path}/processed_data/dev/reviews_users.npy", allow_pickle=True).item()
reviews_movie_dev = np.load(f"{path}/processed_data/dev/reviews_movie.npy", allow_pickle=True).item()


print("DATA LOADING:", time.time()-start, flush=True)

# Download Glove Embeddings
URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
FILE = f"{path}/glove"


def tokenize_and_encode():
    # Tokenize, build vocabulary, encode tokens
    print("Tokenizing...\n")
    max_len = 0

    s = time.time()
    tokenized_texts_train, word2idx, train_labels, max_len = tokenize(comments_train, reviews_grades_train, max_len)
    #tokenized_texts_test, word2idx, max_len = tokenize(comments_test)
    tokenized_texts_dev, word2idx, dev_labels, max_len = tokenize(comments_dev, reviews_grades_dev, max_len)
    print("2 TOKENIZE", time.time()-s, flush=True)

    s = time.time()
    input_ids_dev = encode(tokenized_texts_dev, word2idx, max_len)
    input_ids_train = encode(tokenized_texts_train, word2idx, max_len)
    print("2 ENCODE", time.time()-s, flush=True)

    #input_ids_test = encode(tokenized_texts_test, word2idx, max_len)

    print("input_ids_train = ", input_ids_train[0])

    # Load pretrained vectors
    # tokenized_texts, word2idx, max_len = tokenize(np.concatenate((train_texts, test_texts), axis=None))
    print(" file = ", FILE)
    embeddings = load_pretrained_vectors(word2idx, f"{FILE}/glove.6B.300d.txt")
    embeddings = torch.tensor(embeddings)

    print(" Une partie d'embedding = ", len(embeddings[0]))


    # Load data to PyTorch DataLoader
    train_inputs = input_ids_train
    #test_inputs = input_ids_test
    dev_inputs = input_ids_dev

    #print("TRAIN LABELS", train_labels[0].shape, flush=True)
    #print("TRAIN INPUTS", train_inputs.shape, flush=True)

    #test_labels = notes_test
    train_dataloader, dev_dataloader = data_loader(train_inputs, dev_inputs, train_labels, dev_labels, batch_size=512)

    return train_dataloader, dev_dataloader, len(word2idx)

def initilize_model(pretrained_embedding=None,
                    freeze_embedding=False,
                    vocab_size=None,
                    embed_dim=300,
                    filter_sizes=[3, 4, 5],
                    num_filters=[100, 100, 100],
                    num_classes=10,
                    dropout=0.5,
                    learning_rate=0.01,
                    weight_decay=0):
    """Instantiate a CNN model and an optimizer."""

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
    train_dataloader, dev_dataloader, vocab_size = tokenize_and_encode()
    print("GETTING dataloaders:", time.time()-start, flush=True)

    cnn_rand, optimizer = initilize_model(vocab_size=vocab_size,
                                        embed_dim=300,
                                        learning_rate=0.1,
                                        dropout=0.5,
                                        weight_decay=1e-3)

    cnn_rand = cnn_rand.to(device)


    print("STARTING TRAINING…", flush=True)
    start = time.time()
    best_acc, train_time = train(cnn_rand, optimizer, train_dataloader, dev_dataloader, epochs=50, model_name="mr_cnn_rand")
    print("END OF TRAINING:", time.time()-start, flush=True)

main()
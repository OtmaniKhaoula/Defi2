"""
CNN avec Pytorch pour classification des sentiments
Lien du code: https://www.kaggle.com/code/williamlwcliu/cnn-text-classification-pytorch/notebook
Otmani Khaoula & Martin Elisa
"""

import os
import sys
import torch
import re
import fasttext
import tqdm
import numpy as np
import pandas as pd
import nltk
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
# Training et validation
import random
import time
from nltk.tokenize import word_tokenize
from collections import defaultdict

#nltk.download("all")


directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

for key in comments_train:
    comments_train[key] = " ".join(comments_train[key])

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

"""
Chargement du jeu de donnees de test
"""
"""
comments_test = np.load(f"{path}/processed_data/test/comments.npy", allow_pickle=True).item()
data_test = pd.DataFrame(list(comments_test.items()), columns=['id_reviews', 'comments'])
comments_test = data_test['comments'].values.tolist()
"""

"""
Chargement du jeu de donnees de développement
"""

reviews_grades_dev = np.load(f"{path}/processed_data/dev/reviews_grades.npy", allow_pickle=True).item()
comments_dev = np.load(f"{path}/processed_data/dev/comments.npy", allow_pickle=True).item()
reviews_users_dev = np.load(f"{path}/processed_data/dev/reviews_users.npy", allow_pickle=True).item()
reviews_movie_dev = np.load(f"{path}/processed_data/dev/reviews_movie.npy", allow_pickle=True).item()

for key in comments_dev:
    comments_dev[key] = " ".join(comments_dev[key])


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

# Download Glove Embeddings
URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
FILE = f"{path}/glove"


"""Tokenize texts, build vocabulary and find maximum sentence length.

Args:
    texts (List[str]): List of text data
Returns:
    tokenized_texts (List[List[str]]): List of list of tokens
    word2idx (Dict): Vocabulary built from the corpus
    max_len (int): Maximum sentence length
"""
# tokenization
def tokenize(texts):

    max_len = 0
    tokenized_texts = []
    word2idx = {}

    # Add <pad> and <unk> tokens to the vocabulary
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1

    # Building our vocab from the corpus starting from index 2
    idx = 2
    for sent in texts:
        tokenized_sent = word_tokenize(sent)

        # Add `tokenized_sent` to `tokenized_texts`
        tokenized_texts.append(tokenized_sent)

        # Add new token to `word2idx`
        for token in tokenized_sent:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

        # Update `max_len`
        max_len = max(max_len, len(tokenized_sent))

    return tokenized_texts, word2idx, max_len

"""Pad each sentence to the maximum sentence length and encode tokens to
their index in the vocabulary.

Returns:
    input_ids (np.array): Array of token indexes in the vocabulary with
       shape (N, max_len). It will the input of our CNN model.
"""
def encode(tokenized_texts, word2idx, max_len):

    input_ids = []
    for tokenized_sent in tokenized_texts:
        # Pad sentences to max_len
        tokenized_sent += ['<pad>'] * (max_len - len(tokenized_sent))

        # Encode tokens to input_ids
        input_id = [word2idx.get(token) for token in tokenized_sent]
        input_ids.append(input_id)
    
    return np.array(input_ids)

"""Load pretrained vectors and create embedding layers.

Args:
    word2idx (Dict): Vocabulary built from the corpus
    fname (str): Path to pretrained vector file

Returns:
    embeddings (np.array): Embedding matrix with shape (N, d) where N is
        the size of word2idx and d is embedding dimension
"""
# load embeddings
def load_pretrained_vectors(word2idx, fname):

    print("Loading pretrained vectors...")
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # n, d = map(int, fin.readline().split())
    d=300

    # Initilize random embeddings
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
    embeddings[word2idx['<pad>']] = np.zeros((d,))

    # Load pretrained vectors
    count = 0
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word2idx:
            count += 1
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")

    return embeddings

# Tokenize, build vocabulary, encode tokens
print("Tokenizing...\n")
tokenized_texts_train, word2idx, max_len = tokenize(comments_train)
#tokenized_texts_test, word2idx, max_len = tokenize(comments_test)
tokenized_texts_dev, word2idx, max_len = tokenize(comments_dev)
tokenized_texts, word2idx, max_len = tokenize(np.concatenate((comments_train, comments_dev), axis=None))
#print("tokenized text = ", tokenized_texts[0])
#print("word2idx = ", word2idx) #dictionnaire/vocabulaire
#print("max_len = ", max_len)
input_ids_dev = encode(tokenized_texts_dev, word2idx, max_len)
input_ids_train = encode(tokenized_texts_train, word2idx, max_len)
#input_ids_test = encode(tokenized_texts_test, word2idx, max_len)

print("input_ids_train = ", input_ids_train[0])

# Load pretrained vectors
# tokenized_texts, word2idx, max_len = tokenize(np.concatenate((train_texts, test_texts), axis=None))
print(" file = ", FILE)
embeddings = load_pretrained_vectors(word2idx, f"{FILE}/glove.6B.300d.txt")
embeddings = torch.tensor(embeddings)

print(" Une partie d'embedding = ", len(embeddings[0]))

# DataLoader
def data_loader(train_inputs, dev_inputs, train_labels, dev_labels,batch_size=512):
    # Convert data type to torch.Tensor
    train_inputs, dev_inputs, train_labels, dev_labels =\
    tuple(torch.tensor(data).to(device) for data in
          [train_inputs, dev_inputs, train_labels, dev_labels])

    # Specify batch_size
    batch_size = batch_size

    # Create DataLoader for training data
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data
    dev_data = TensorDataset(dev_inputs, dev_labels)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)
    """
    # Create DataLoader for test data
    test_data = TensorDataset(test_inputs, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    """
    return train_dataloader, dev_dataloader

# Load data to PyTorch DataLoader
train_inputs = input_ids_train
#test_inputs = input_ids_test
dev_inputs = input_ids_dev
train_labels = notes_train
#test_labels = notes_test
dev_labels = notes_dev
train_dataloader, dev_dataloader = data_loader(train_inputs, dev_inputs, train_labels, dev_labels, batch_size=512)

# Création du modèle
class CNN_NLP(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=10,
                 dropout=0.5):
        """
        The constructor for CNN_NLP class.

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. Default: False
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            n_classes (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """

        super(CNN_NLP, self).__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))
        #print("logits = ", logits)

        return logits

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

# Specify loss function
loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    """Set seed for reproducibility."""

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, optimizer, train_dataloader, dev_dataloader=None, epochs=50, model_name=""):
    """Train the CNN model."""
    
    # Tracking best validation accuracy
    best_accuracy = 0
    train_time = 0

    # Start training loop
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'dev Loss':^10} | {'dev Acc':^9} | {'Elapsed':^9}")
    print("-"*60)

    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0

        # Put the model into the training mode
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Load batch to GPU
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            # print("loss train = ", loss)
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        train_time += time.time() - t0_epoch

        # =======================================
        #               Validation
        # =======================================
        if dev_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            dev_loss, dev_accuracy = evaluate(model, dev_dataloader)
                       
            # Track the best accuracy
            if dev_accuracy > best_accuracy:
                best_accuracy = dev_accuracy
                torch.save(model, "./models/" + model_name + "_best_model.pt")

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {dev_loss:^10.6f} | {dev_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            
    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")
    return best_accuracy, train_time

def evaluate(model, test_dataloader):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    test_accuracy = []
    test_loss = []

    # For each batch in our validation set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        # print("loss test = ", loss)
        test_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        test_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    test_loss = np.mean(test_loss)
    test_accuracy = np.mean(test_accuracy)

    return test_loss, test_accuracy

# Create a directory named 'models' if it doesn't exist
os.makedirs('./models', exist_ok=True)

# need to create dir: "./models"
# CNN-rand: Word vectors are randomly initialized.
# set_seed(42)
cnn_rand, optimizer = initilize_model(vocab_size=len(word2idx),
                                      embed_dim=300,
                                      learning_rate=0.1,
                                      dropout=0.5,
                                      weight_decay=1e-3)

cnn_rand = cnn_rand.to(device)

best_acc, train_time = train(cnn_rand, optimizer, train_dataloader, dev_dataloader, epochs=20, model_name="mr_cnn_rand")

import os, sys
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)

directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(directory)

import config
path = config.paths['url']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, train_dataloader, dev_dataloader=None, epochs=50, model_name=""):
    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()

    # Tracking best validation accuracy
    best_accuracy = 0
    train_time = 0

    # Start training loop
    print("Start training...\n", flush=True)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'dev Loss':^10} | {'dev Acc':^9} | {'Elapsed':^9}", flush=True)
    print("-"*60, flush=True)

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
            #print("TRAIN preds", torch.argmax(logits, dim=1), flush=True)
            #print("REAL VAL", b_labels, flush=True)
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
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {dev_loss:^10.6f} | {dev_accuracy:^9.2f} | {time_elapsed:^9.2f}", flush=True)
            
    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")
    return best_accuracy, train_time

def evaluate(model, test_dataloader):
    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()

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
        #print(preds, flush=True)
        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        test_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    test_loss = np.mean(test_loss)
    test_accuracy = np.mean(test_accuracy)

    return test_loss, test_accuracy

def predictions(model, test_dataloader):
    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()

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

    return preds

"""Tokenize texts, build vocabulary and find maximum sentence length.

Args:
    texts (List[str]): List of text data
Returns:
    tokenized_texts (List[List[str]]): List of list of tokens
    word2idx (Dict): Vocabulary built from the corpus
    max_len (int): Maximum sentence length
"""
# tokenization
def tokenize(texts, grades, max_len, test=False):

    tokenized_texts = []
    labels = []

    word2idx = {}

    # Add <pad> and <unk> tokens to the vocabulary
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1

    # Building our vocab from the corpus starting from index 2
    idx = 2
    for key in texts:
        if not texts[key] == "":
            if not test:
                labels.append(int(float(grades[key])*2-1))
            else:
                labels.append(0)

            tokenized_texts.append(texts[key])
            for word in texts[key]:
                if not word in word2idx:
                    word2idx[word] = idx
                    idx += 1
            
            max_len = max(max_len, len(texts[key]))

    return tokenized_texts, word2idx, labels, max_len

"""Pad each sentence to the maximum sentence length and encode tokens to
their index in the vocabulary.

Returns:
    input_ids (np.array): Array of token indexes in the vocabulary with
       shape (N, max_len). It will the input of our CNN model.
"""
def encode(tokenized_texts, word2idx, max_len):

    input_ids = [
        [word2idx.get(token, word2idx['<pad>']) for token in tokens] + [word2idx['<pad>']] * (max_len - len(tokens))
        for tokens in tokenized_texts
    ]

    return np.array(input_ids)

"""Load pretrained vectors and create embedding layers.

Args:
    word2idx (Dict): Vocabulary built from the corpus
    fname (str): Path to pretrained vector file

Returns:
    embeddings (np.array): Embedding matrix with shape (N, d) where N is
        the size of word2idx and d is embedding dimension
"""

def tokenize_and_encode(comments, grades, max_len, test=False):
    # Download Glove Embeddings
    #URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
    FILE = f"{path}/glove"


    # Tokenize, build vocabulary, encode tokens
    print("Tokenizing...\n")

    tokenized_texts, word2idx, labels, max_len = tokenize(comments, grades, max_len, test=test)

    input_ids = encode(tokenized_texts, word2idx, max_len)

    # Load pretrained vectors
    embeddings = load_pretrained_vectors(word2idx, f"{FILE}/glove.6B.300d.txt")
    embeddings = torch.tensor(embeddings)

    # Load data to PyTorch DataLoader
    inputs = input_ids


    dataloader = data_loader(inputs, labels, batch_size=128)

    return dataloader, len(word2idx)
    
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
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word2idx:
            count += 1
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")

    return embeddings

# DataLoader
def data_loader(inputs, labels, batch_size=128):
    # Convert data type to torch.Tensor
    inputs, labels = tuple(torch.tensor(data).to(device) for data in
          [inputs, labels])

    # Specify batch_size
    batch_size = batch_size

    # Create DataLoader 
    data = TensorDataset(inputs, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader

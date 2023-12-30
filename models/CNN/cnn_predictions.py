import torch
import torch.nn as nn
import os, sys
import numpy as np
from utils import predictions, tokenize_and_encode

directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(directory)

import config
path = config.paths['url']

if torch.cuda.is_available():
    print("GPU disponible!")
else:
    print("Aucun GPU disponible. Vérifiez votre configuration.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

force_cudnn_initialization()

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

comments_test = np.load(f"{path}/processed_data/test/comments_clean.npy", allow_pickle=True).item()
metadata_test = np.load(f"{path}/processed_data/test/movie_metadata.npy", allow_pickle=True).item()
review_movie_test = np.load(f"{path}/processed_data/test/reviews_movie.npy", allow_pickle=True).item()
reviews_users_test = np.load(f"{path}/processed_data/test/reviews_users.npy", allow_pickle=True).item()

for key in comments_train:
    movie = review_movie_train[key]

    if not comments_train[key] == "" and not reviews_users_train[key] == "":
        comments_train[key] = comments_train[key] + [reviews_users_train[key]] + metadata_train[movie].split(" ") + [movie]


for key in comments_dev:
    movie = review_movie_dev[key]
    
    if not comments_dev[key] == "" and not reviews_users_dev[key] == "":
        comments_dev[key] = comments_dev[key] + [reviews_users_dev[key]] + metadata_dev[movie].split(" ") + [movie]


for key in comments_test:
    movie = review_movie_test[key]
    
    if not comments_test[key] == "" and not reviews_users_test[key] == "":
        comments_test[key] = comments_test[key] + [reviews_users_test[key]] + metadata_test[movie].split(" ") + [movie]
    else:
        comments_test[key] = [comments_test[key]] + [reviews_users_test[key]] + metadata_test[movie].split(" ") + [movie]

max_len = 2677
cnn_rand = torch.load(f"{path}/models/CNN/models/lr=0,8-drop0,25-emb-dim=300-all-meta_best_model.pt")
cnn_rand = cnn_rand.to(device)


print("generating train dataloader…", flush=True)
word2idx = {}

train_dataloader, vocab_size, word2idx, _ = tokenize_and_encode(comments_train, reviews_grades_train, max_len, word2idx)
dev_dataloader, vocab_size, word2idx, _ = tokenize_and_encode(comments_dev, reviews_grades_dev, max_len, word2idx)
test_dataloader, vocab_size, word2idx, key_dict = tokenize_and_encode(comments_test, {}, max_len, word2idx, test=True)

print("STARTING TEST PREDS…", flush=True)
preds, labels = predictions(cnn_rand, test_dataloader)

ids = list(comments_test.keys())

to_save = ""

for i in range(len(preds)):
    for j in range(len(preds[i])):
        grade = preds[i][j] + 1
        to_save += f"{key_dict[labels[i][j]]} {str(float(grade)/float(2)).replace('.', ',')}\n"
        
target = open("../../predictions/CNN-predictions.txt", "w")
target.write(to_save)
target.close()

print("finish", flush=True)
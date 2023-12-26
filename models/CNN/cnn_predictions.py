import torch
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


comments_test = np.load(f"{path}/processed_data/test/comments.npy", allow_pickle=True).item()

max_len = 2677
test_dataloader, vocab_size = tokenize_and_encode(comments_test, {}, max_len, test=True)

cnn_rand = torch.load(f"{path}/models/CNN/models/mr_cnn_rand_best_model-ALLDATA.pt")
cnn_rand = cnn_rand.to(device)

preds = predictions(cnn_rand, test_dataloader)
print(preds)


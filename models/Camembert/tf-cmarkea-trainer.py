from datasets import DatasetDict, Dataset
import torch
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer, AutoConfig
from transformers import AutoModelForSequenceClassification, CamembertForSequenceClassification
import evaluate
import sys, os

directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(directory)

import config
path = config.paths['url']

print("TF-BLARD META DATA 1e5", flush=True)
writer = SummaryWriter("tfblad-1e5-meta")

if torch.cuda.is_available():
    print("GPU disponible!")
    torch.cuda.empty_cache()
else:
    print("Aucun GPU disponible. Vérifiez votre configuration.")

# Vérifier la disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

# Données apprentissage

reviews_grades_train = np.load(f"{path}/processed_data/train/reviews_grades.npy", allow_pickle=True).item()
reviews_grades_train = {key: value * 2 - 1 for key, value in reviews_grades_train.items()}
comments_train = np.load(f"{path}/processed_data/train/comments_meta.npy", allow_pickle=True).item()
df_comments_train = pd.DataFrame(list(comments_train.items()), columns=['id_reviews', 'text'])
df_reviews_grades_train = pd.DataFrame(list(reviews_grades_train.items()), columns=['id_reviews', 'labels'])
df_reviews_grades_train["labels"] = df_reviews_grades_train["labels"].astype(int)
data_train = pd.merge(df_comments_train, df_reviews_grades_train, on='id_reviews')
#data_train = data_train.groupby('labels').apply(lambda x: x.sample(n=150)).reset_index(drop=True)

# Assuming df is your DataFrame
# Replace whitespace values with NaN
data_train.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

# Drop rows with NaN values
data_train.dropna(inplace=True)

# Drop rows with empty strings
data_train = data_train[data_train.astype(str).applymap(lambda x: x.strip() != "")]

# Reset index if needed
data_train.reset_index(drop=True, inplace=True)

#data_train = data_train.groupby('labels').apply(lambda x: x.sample(n=15)).reset_index(drop=True)

# Données dev
reviews_grades_dev = np.load(f"{path}/processed_data/dev/reviews_grades.npy", allow_pickle=True).item()
reviews_grades_dev = {key: value * 2 - 1 for key, value in reviews_grades_dev.items()}
comments_dev = np.load(f"{path}/processed_data/dev/comments_meta.npy", allow_pickle=True).item()
df_comments_dev = pd.DataFrame(list(comments_dev.items()), columns=['id_reviews', 'text'])
df_reviews_grades_dev = pd.DataFrame(list(reviews_grades_dev.items()), columns=['id_reviews', 'labels'])
df_reviews_grades_dev["labels"] = df_reviews_grades_dev["labels"].astype(int)
data_dev = pd.merge(df_comments_dev, df_reviews_grades_dev, on='id_reviews')

# Assuming df is your DataFrame
# Replace whitespace values with NaN
data_dev.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

# Drop rows with NaN values
data_dev.dropna(inplace=True)

# Drop rows with empty strings
data_dev = data_dev[data_dev.astype(str).applymap(lambda x: x.strip() != "")]

# Reset index if needed
data_dev.reset_index(drop=True, inplace=True)

#data_dev = data_dev.groupby('labels').apply(lambda x: x.sample(n=150)).reset_index(drop=True)

train_dataset = Dataset.from_pandas(data_train[['text', 'labels']])
dev_dataset = Dataset.from_pandas(data_dev[['text', 'labels']])

dataset_dict = DatasetDict({"train": train_dataset, "dev": dev_dataset})
print("dictionnaire de dataset = ", dataset_dict)

# Tokenisation
from transformers import AutoTokenizer
model_name = "tblard/tf-allocine" #"cmarkea/distilcamembert-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"]
dev_dataset = tokenized_datasets["dev"]
print(type(dev_dataset))

config = AutoConfig.from_pretrained(model_name)
config.num_labels = 10
model = AutoModelForSequenceClassification.from_config(config)

#model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=10, ignore_mismatched_sizes=True)

output_d = f"{path}/tblardtf-1e5-meta/"
print("OUTPUT DIR:", output_d, flush=True)

training_args = TrainingArguments(
    output_dir=output_d,
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=1e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="tensorboard",
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

import torch
torch.cuda.empty_cache()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(f"path/tfblad-1e5-meta/")
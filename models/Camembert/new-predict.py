from datasets import DatasetDict, Dataset
import torch
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
import evaluate
import sys, os

directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(directory)

import config
path = config.paths['url']

writer = SummaryWriter("test_trainer")

#f_pred_name = "../../predictions/cmarkea-all-chp-10406.txt"
f_pred_name = "../../predictions/camembert-da-4.txt"
print(f_pred_name)

if torch.cuda.is_available():
    print("GPU disponible!")
else:
    print("Aucun GPU disponible. Vérifiez votre configuration.")

# Vérifier la disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

# Données apprentissage

reviews_grades_test = np.load(f"{path}/processed_data/test/reviews_grades.npy", allow_pickle=True).item()
test_keys = np.load(f"{path}/processed_data/test/keys.npy", allow_pickle=True).item()
#reviews_grades_test = {key: value * 2 - 1 for key, value in reviews_grades_test.items()}
comments_test = np.load(f"{path}/processed_data/test/comments_meta2.npy", allow_pickle=True).item()
df_comments_test = pd.DataFrame(list(comments_test.items()), columns=['id_reviews', 'text'])
df_reviews_grades_test = pd.DataFrame(list(reviews_grades_test.items()), columns=['id_reviews', 'keys'])
df_reviews_grades_test["keys"] = df_reviews_grades_test["keys"].astype(int)
data_test = pd.merge(df_comments_test, df_reviews_grades_test, on='id_reviews')

"""# Assuming df is your DataFrame
# Replace whitespace values with NaN
data_test.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

# Drop rows with NaN values
data_test.dropna(inplace=True)

# Drop rows with empty strings
data_test = data_test[data_test.astype(str).applymap(lambda x: x.strip() != "")]

# Reset index if needed
data_test.reset_index(drop=True, inplace=True)

data_test = data_test.groupby('labels').apply(lambda x: x.sample(n=32)).reset_index(drop=True)"""

# Données dev
reviews_grades_dev = np.load(f"{path}/processed_data/train/reviews_grades.npy", allow_pickle=True).item()
reviews_grades_dev = {key: value * 2 - 1 for key, value in reviews_grades_dev.items()}
comments_dev = np.load(f"{path}/processed_data/train/comments_meta.npy", allow_pickle=True).item()

"""l = 0
for key in comments_dev:
    print("lo", comments_dev[key], flush=True)
    if l == 10:
        break
    l += 1
exit()"""

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

data_dev = data_dev.groupby('labels').apply(lambda x: x.sample(n=150)).reset_index(drop=True)

test_dataset = Dataset.from_pandas(data_test[['text']])
dev_dataset = Dataset.from_pandas(data_dev[['text', 'labels']])

dataset_dict = DatasetDict({"test": test_dataset, "dev": dev_dataset})
print("dictionnaire de dataset = ", dataset_dict)

# Tokenisation
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("camembert-base")

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

test_dataset = tokenized_datasets["test"]
dev_dataset = tokenized_datasets["dev"]
print(type(dev_dataset))

model = AutoModelForSequenceClassification.from_pretrained(f"{path}/test_trainer-all-1e5-da2/checkpoint-56500/", num_labels=10, local_files_only=True)

training_args = TrainingArguments(
    output_dir="test_trainer",
    num_train_epochs=5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=1e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="tensorboard",
    hub_strategy="checkpoint",
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
    #train_dataset=test_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

predict = trainer.predict(test_dataset)
print("predict:", predict, flush=True)
predictions = np.argmax(predict.predictions, axis=-1)
print("predictions:", predictions, flush=True)
#print("labels:", predict.label_ids.shape, flush=True)

to_save = ""

for i in range(len(predictions)):
    grade = predictions[i] + 1
    to_save += f"{df_comments_test['id_reviews'][i]} {str(float(grade)/float(2)).replace('.', ',')}\n"
        
target = open(f_pred_name, "w")
target.write(to_save)
target.close()

print("finish", flush=True)

import sys, os
import numpy as np
from transformers import pipeline

directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(directory)

import config
path = config.paths['url']

comments_test = np.load(f"{path}/processed_data/train/comments.npy", allow_pickle=True).item()
rev_grades = np.load(f"{path}/processed_data/train/reviews_grades.npy", allow_pickle=True).item()

a = 0
for el in comments_test:
    print("id, grade", el, rev_grades[el])
    a += 1
    if a > 10:
        break

analyzer = pipeline(
    task='text-classification',
    model="cmarkea/distilcamembert-base-sentiment",
    tokenizer="cmarkea/distilcamembert-base-sentiment"
)

commentaires = [comment for comment in comments_test.values()]
review_ids = [ids for ids in comments_test.keys()]
grades = [grade for grade in rev_grades.values()]


for i in range(10):
    result = analyzer(
        commentaires[i],
        top_k=None,
        truncation=True, 
    )
    note=0

    for j in range(len(result)): 
        note += float(result[j]['score'])*(int(result[j]['label'].split(" star")[0])-1)
        
    print(review_ids[i], note, grades[i], flush=True)

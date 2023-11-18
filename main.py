import fasttext
import numpy as np
import tqdm

rev_comments = np.load("processed_data/test/comments.npy", allow_pickle=True).item()

model = fasttext.train_supervised(input="processed_data/train/data.tsv")
#model.test("processed_data/dev/data.tsv")

results = []

for key in tqdm.tqdm(rev_comments):
    pred = model.predict(rev_comments[key])
    results.append(key+" "+pred[0][0].split('__label__')[1].replace('.', ',')+"\n")

f = open(f'predictions/prediction.txt', 'w', newline='')
f.writelines(results)

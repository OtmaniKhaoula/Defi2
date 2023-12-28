import re
import string 
import fasttext
import numpy as np
import tqdm
import os, sys

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)

import config

path = config.paths['url']
test_comments = np.load(f"{path}/processed_data/test/comments.npy", allow_pickle=True).item()
train_comments = np.load(f"{path}/processed_data/train/comments.npy", allow_pickle=True).item()
train_grades = np.load(f"{path}/processed_data/train/reviews_grades.npy", allow_pickle=True).item()

words_freq = {}
words_id = {}
n = 1

"""def readfile(file):
    with open(file) as f:
        content = f.read()

    return content.split("⁠******")[2][2:]

def pre_processing(content):
    content = ponctuation(content)
    tweets = tokenize(content)
    #tweets = stemming(tweets)

    return tweets


def tokenize(content):
    tweets = []

    for key in content:
        tweets.append(c.split("\t"))
        print(tweets)
        exit()

        one_gram = tweets[-1][1].split(" ")
        tweets[-1][1] = []

        if(n > 1):
            for i in range(0, len(one_gram)-n, n):
                tweets[-1][1].append(one_gram[i] + " " + one_gram[i+1])
        else:
            for i in range(len(one_gram)):
                tweets[-1][1].append(one_gram[i])

        tweets[-1][1] = [t.lower() for t in tweets[-1][1]]

    return tweets"""

def set_words_freq(tweets):
    for key in tweets:
        for word in tweets[key]:
            if word in words_freq:
                words_freq[word] = words_freq[word]+1
            else:
                words_freq[word] = 1

def set_words_id():
    i = 1
    for key in words_freq:
        words_id[key] = i
        i += 1

def ponctuation(content):
    content = content.translate(str.maketrans('', '', string.punctuation)).replace("\\u2019", "").replace("\\u002", "")
    content = re.sub('http://\S+|https://\S+', '', content)
    content = re.sub('\d', '', content)

    return content.splitlines()

def stemming(tweets):
    for j in range(len(tweets)):
        for i in range(len(tweets[j][1])):
            if len(tweets[j][1][i]) > 4:
                tweets[j][1][i] = tweets[j][1][i][:4]
    return tweets

def get_word_id(word):
    if word in words_id:
        return words_id[word]
    return 0

def generate_svm_file(tweets, name):
    f = open("./SVMs/"+name, "w")

    for key in tweets:
        tweet = tweets[key]
        words = sorted(set(tweet))
        words.sort(key=get_word_id)

        grade = str(float(train_grades[key])*2)
        write_label(f, grade)
        for word in words:
            if word in words_id:
                f.write(" " + str(words_id[word]) + ":" + str(tweet.count(word)))
        f.write("\n")
    f.close()

def generate_test_svm_file(tweets, name):
    f = open("./SVMs/"+name, "w")
    ids = []
    for key in tweets:
        ids.append(key)
        tweet = tweets[key]
        write_label(f, 0)
        words = sorted(set(tweet.split(' ')))
        words.sort(key=get_word_id)

        for word in words:
            if word in words_id:
                f.write(" " + str(words_id[word]) + ":" + str(words.count(word)))
        f.write("\n")
    f.close()

    fi = open("ids.txt", "w")
    for i in ids:
        fi.write(str(i)+"\n")
    fi.close


def write_label(f, lab):
    f.write(str(lab))

def main():
    tweets = train_comments #tokenize(train_comments)
    set_words_freq(tweets)
    set_words_id()
    generate_svm_file(tweets, "train.svm")
    print(len(words_id))

    tweets = test_comments
    generate_test_svm_file(tweets, "test.svm")
main()
import numpy as np
import xml.etree.ElementTree as ET
import data_processing
import time
import sys, os

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)
print("DIRECTORY", directory)

import config

path = config.paths['url']

file = sys.argv[1] #file to read train, dev or test

tree = ET.parse(f"{path}/data/{file}.xml")
root = tree.getroot()

reviews_grades = {} #dictionnaire des review_id -> notes
reviews_users = {}
reviews_movie = {}
movie_grades = {}
corpus = {}

def gen_dicts(test=False):
    for comment in root.findall("comment"):
        review_id = comment.find('review_id').text
        if not test:
            note = float(comment.find('note').text.replace(',', '.'))
        commentaire = comment.find('commentaire').text
        movie_id = comment.find('movie').text
        user_id = comment.find('user_id').text


        #map review with it's grade
        if not test:
            reviews_grades[review_id] = note

            #map movie with grades
            if movie_id in movie_grades:
                movie_grades[movie_id].append(note)
            else:
                movie_grades[movie_id] = [note]

        reviews_movie[review_id] = movie_id
        reviews_users[review_id] = user_id
        corpus[review_id] = commentaire


    cleaned_corpus = data_processing.preprocessing_text(corpus)

    if test:
        reviews_grades = data_processing.gen_test_review_grades(corpus)
    #fast_text = data_processing.preprocessing_fasttext(corpus, reviews_grades, reviews_users, reviews_movie, file)

    np.save(f"{path}/processed_data/{file}/movie_grades.npy", movie_grades)
    np.save(f"{path}/processed_data/{file}/reviews_movie.npy", reviews_movie)
    np.save(f"{path}/processed_data/{file}/reviews_grades.npy", reviews_grades)
    np.save(f"{path}/processed_data/{file}/reviews_users.npy", reviews_users)
    np.save(f"{path}/processed_data/{file}/comments_clean.npy", cleaned_corpus)
    np.save(f"{path}/processed_data/{file}/comments.npy", corpus)


def gen_test_corpus():
    for comment in root.findall("comment"):
        review_id = comment.find('review_id').text
        commentaire = comment.find('commentaire').text
       
        corpus[review_id] = commentaire

    cleaned_corpus = data_processing.preprocessing_test(corpus)
    np.save(f"{path}/processed_data/{file}/comments.npy", cleaned_corpus)


if not file == 'test':
    gen_dicts()
else:
    gen_dicts(test=True)
    gen_test_corpus()

import numpy as np
import xml.etree.ElementTree as ET
import data_processing
import time
import sys

file = sys.argv[1] #file to read
to_save_folder = sys.argv[2] #fodler to save the the produced files

tree = ET.parse(file)
root = tree.getroot()

reviews_grades = {} #dictionnaire des review_id -> notes
reviews_users = {}
reviews_movie = {}
movie_grades = {}
corpus = {}

def gen_dicts():
    for comment in root.findall("comment"):
        review_id = comment.find('review_id').text
        note = float(comment.find('note').text.replace(',', '.'))
        commentaire = comment.find('commentaire').text
        movie_id = comment.find('movie').text
        user_id = comment.find('user_id').text


        #map review with it's grade
        reviews_grades[review_id] = note
        reviews_movie[review_id] = movie_id
        reviews_users[review_id] = user_id
        corpus[review_id] = commentaire

        #map movie with grades
        if movie_id in movie_grades:
            movie_grades[movie_id].append(note)
        else:
            movie_grades[movie_id] = [note]

    cleaned_corpus = data_processing.preprocessing_text(corpus)
    fast_text = data_processing.preprocessing_fasttext(corpus, reviews_grades, to_save_folder)

    np.save(f"../processed_data/{to_save_folder}/movie_grades.npy", movie_grades)
    np.save(f"../processed_data/{to_save_folder}/reviews_movie.npy", reviews_movie)
    np.save(f"../processed_data/{to_save_folder}/reviews_grades.npy", reviews_grades)
    np.save(f"../processed_data/{to_save_folder}/reviews_users.npy", reviews_users)
    np.save(f"../processed_data/{to_save_folder}.npy", cleaned_corpus)


gen_dicts()

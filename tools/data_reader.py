import numpy as np
import xml.etree.ElementTree as ET
import data_processing

tree = ET.parse('../data/train.xml')
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

    np.save("../processed_data/movie_grades.npy", movie_grades)
    np.save("../processed_data/reviews_movie.npy", movie_grades)
    np.save("../processed_data/reviews_grades.npy", reviews_grades)
    np.save("../processed_data/reviews_users.npy", reviews_users)
    np.save("../processed_data/corpus.npy", cleaned_corpus)

gen_dicts()

import operator
import numpy as np
import graphics
import data_processing
#from wordcloud import WordCloud

reviews_grades = np.load("../processed_data/reviews_grades.npy", allow_pickle=True).item()
reviews_users = np.load("../processed_data/reviews_users.npy", allow_pickle=True).item()
movie_grades = np.load("../processed_data/movie_grades.npy", allow_pickle=True).item()
corpus = np.load("../processed_data/corpus.npy", allow_pickle=True).item()

def get_mean_grades():
    mean_grade = 0
    for key in reviews_grades:
        mean_grade += reviews_grades[key]

    mean_grade = mean_grade/len(reviews_grades)
    return mean_grade

def get_mean_grades_by_movie():
    mean_grades_by_movies = {}
    for key in movie_grades:
        mean_grades_by_movies[key] = sum(movie_grades[key])/len(movie_grades[key])

    return mean_grades_by_movies

def get_grades_repartition():
    grade_repartition = {}
    for value in np.arange(0.5, 5.5, 0.5):
        grade_repartition[value] = operator.countOf(reviews_grades.values(), value)
    return grade_repartition

def get_grades_repartition_by_movie():
    grades_repartition_by_movie = {}

    for key in movie_grades:
        if not key in grades_repartition_by_movie:
            grades_repartition_by_movie[key] = []

        for value in np.arange(0.5, 5.5, 0.5):       
            grades_repartition_by_movie[key].append((movie_grades[key].count(value), value))
        
    return grades_repartition_by_movie

def get_grades_by_user():
    grades_by_user = {}
    for key in reviews_users:
        if not reviews_users[key] in grades_by_user:
            grades_by_user[reviews_users[key]] = []
        
        grades_by_user[reviews_users[key]].append(reviews_grades[key])
        
    return grades_by_user 

def get_grades_repartition_by_user(grades_by_user):
    grades_repartition_by_user = {}

    for key in reviews_users:
        if not key in grades_repartition_by_user:
            grades_repartition_by_user[reviews_users[key]] = []

        for value in np.arange(0.5, 5.5, 0.5):       
            grades_repartition_by_user[reviews_users[key]].append((grades_by_user[reviews_users[key]].count(value), value))
        
    return grades_repartition_by_user

#def word_cloud():
    

if __name__ == "__main__":
    mean_note_by_movie = get_mean_grades_by_movie()
    distrib_notes = get_grades_repartition()
    distrib_notes_by_movie = get_grades_repartition_by_movie()
    grades_by_user = get_grades_by_user()
    distrib_notes_by_user = get_grades_repartition_by_user(grades_by_user)

    # Répartition des notes (données apprentissage)
    graphics.graph_repartition(distrib_notes, "notes (données apprentissage)")
    graphics.graph_repartition_by(distrib_notes_by_movie, 4, " films (données d'apprentissage)")
    graphics.graph_repartition_by(distrib_notes_by_user, 4, " utilisateurs (données d'apprentissage)")
    
    new_corpus = data_processing.preprocessing_text(corpus)
    
    freq = graphics.frequence(new_corpus)
    
    keys = list(corpus.keys())[0:3000]
    dictionnaire = {}
    for key in keys:
        dictionnaire[key] = corpus[key]
    new_dictionnaire = data_processing.preprocessing_text(dictionnaire)
        
    freq = graphics.frequence(new_dictionnaire)
    graphics.graph_repartition(freq, "mots les plus fréquents (données apprentissage)")
    
    print("finished")


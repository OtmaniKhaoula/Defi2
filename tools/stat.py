import operator
import numpy as np
import graphics

reviews_grades = np.load("processed_data/reviews_grades.npy", allow_pickle=True).item()
reviews_users = np.load("processed_data/reviews_users.npy", allow_pickle=True).item()
movie_grades = np.load("processed_data/movie_grades.npy", allow_pickle=True).item()
corpus = np.load("processed_data/corpus.npy", allow_pickle=True).item()


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

if __name__ == "__main__" 
    mean_note_by_movie = get_mean_grades_by_movie()
    distrib_notes = get_grades_repartition()
    distrib_notes_by_movie = get_grades_repartition_by_movie()

    # Répartition des notes (données apprentissage)
    graphics.graph_repartition_note(distrib_notes)
    graphics.graph_repartition_by_movie(distrib_notes_by_movie, 4)
    print("finished")


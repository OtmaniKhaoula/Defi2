import xml.etree.ElementTree as ET
import operator
import numpy as np

tree = ET.parse('data/train.xml')
root = tree.getroot()

reviews_grades = {} #dictionnaire des review_id -> notes
#enregistrer dico sur dd pour les recharger + rapdiement (voir numpy)
movie_grades= {}
corpus = {}

def gen_dicts():
    for comment in root.findall("comment"):
        note = float(comment.find('note').text.replace(',', '.'))
        review_id = comment.find('review_id').text
        movie_id = comment.find('movie').text

        #map review with it's grade
        reviews_grades[review_id] = note
    
        #map movie with grades
        if movie_id in movie_grades:
            movie_grades[movie_id].append(note)
        else:
            movie_grades[movie_id] = [note]
        
        print(movie_grades)


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
        for value in np.arange(0.5, 5.5, 0.5):
            grades_repartition_by_movie[key] = movie_grades[key], value

    return grades_repartition_by_movie

def main():
    gen_dicts()
    #get_mean_grades_by_movie()
    #print(get_grades_repartition())
    #print(get_grades_repartition_by_movie())

main()
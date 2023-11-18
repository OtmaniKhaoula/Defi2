import operator
import numpy as np
import graphics
import data_processing
import time
import scipy
import sys, os

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)

import config

path = config.paths['url']
folder_to_load = sys.argv[1]

reviews_grades = np.load(f"{path}/processed_data/{folder_to_load}/reviews_grades.npy", allow_pickle=True).item()
reviews_users = np.load(f"{path}/processed_data/{folder_to_load}/reviews_users.npy", allow_pickle=True).item()
movie_grades = np.load(f"{path}/processed_data/{folder_to_load}/movie_grades.npy", allow_pickle=True).item()
corpus = np.load(f"{path}/processed_data/{folder_to_load}/comments.npy", allow_pickle=True).item()

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

def frequence(corpus):
    words_freq = {}
    for key in corpus.keys():
        for token in corpus[key]:
            if not token in words_freq:
                words_freq[token] = 1
            else:
                words_freq[token] += 1
    
    return words_freq   

# Statistiques sur les notes pour chaque film / utilisateurs
def stat_et_mean(grades):
    dict_stat = {}
    for key in grades:
        dict_stat[key] = {}
        dict_stat[key]["mean"] = np.mean(grades[key])
        dict_stat[key]["std"] = np.std(grades[key])
        dict_stat[key]["nb_note"] = len((grades[key]))
        dict_stat[key]["min"] = np.min(grades[key])
        dict_stat[key]["max"] = np.max(grades[key])

    return dict_stat

def length_comments(corpus):
    length_word = {}
    length_char = {}
    for key in corpus.keys():
        length_word[key] = len(corpus[key])
        length_char[key] = len(" ".join(corpus[key]))
    return length_char, length_word

# Corrélation de Pearson (entre deux variables quanti)
def corr_quant(reviews_grades, length_comments):
    notes = []
    lengths = []
    for key in reviews_grades:
        notes.append(reviews_grades[key])
        lengths.append(length_comments[key])
    # Calcul corrélation     
    pearson = scipy.stats.pearsonr(notes, lengths).pvalue

    return pearson

def most_frequent_words_and_their_mean_grade(reviews_grades, corpus):
    mean_grade_by_words = {}

    for key in corpus:
        for word in corpus[key]:
            if word not in mean_grade_by_words:
                mean_grade_by_words[word] = [word, reviews_grades[key], 1]
            else:
                mean_grade_by_words[word][2] += 1
                mean_grade_by_words[word][1] += reviews_grades[key]
    
    for key in mean_grade_by_words:
        mean_grade_by_words[key][1] = mean_grade_by_words[key][1]/mean_grade_by_words[key][2]
    
    mean_grade_by_words = dict(sorted(mean_grade_by_words.items(), key=lambda item: item[1][2], reverse=True))

    return mean_grade_by_words

# Récupérer les commentaires et les notes et récupérer les stats pour les 10 mots les plus fréquents 
def stats_frequent_words(reviews_grades, corpus, list_words):
    notes_by_words = {}

    for key in corpus:
        for words in list_words:
            if(notes_by_words[words] is None):
                notes_by_words[words] = 0
            if(words in corpus[key]):
                notes_by_words[words] += 1

    return notes_by_words


if __name__ == "__main__":
    a = time.time()
    ####### Generating data structures #######
    mean_note_by_movie = get_mean_grades_by_movie()
    distrib_notes = get_grades_repartition()
    distrib_notes_by_movie = get_grades_repartition_by_movie()
    grades_by_user = get_grades_by_user()
    distrib_notes_by_user = get_grades_repartition_by_user(grades_by_user)
    b = time.time()
    print("TIME:", b-a)

    a = time.time()
   
    ######## Statistiques ########
    comments_char, comments_words = length_comments(corpus)
    p = corr_quant(reviews_grades, comments_words)

    print("Longueur moyenne d'un commentaire (mots)", np.mean(list(comments_words.values())))
    print("Longueur moyenne d'un commentaire (charactères)", np.mean(list(comments_char.values())))
    
    print("Pearson coefficient correlation for comment lenths / grade: ", p)
    
    print("Most frequent words and their mean grade: ")
    mean_grade_by_words = most_frequent_words_and_their_mean_grade(reviews_grades, corpus)

    i = 0
    for key in mean_grade_by_words:
        if i == 10:
            break
        
        print(mean_grade_by_words[key])
        i+= 1

    ######## Generating graphs ########
    graphics.graph_repartition(distrib_notes, "notes (données apprentissage)")

    k = list(distrib_notes_by_movie.keys())[0]
    k2 = list(distrib_notes_by_user.keys())[0]
    graphics.graph_boxplot(distrib_notes_by_movie[k], 4, f"distribution des notes pour le film: {k}","boxplot_films_train")
    graphics.graph_boxplot(distrib_notes_by_user[k2], 4, f"distribution des notes pour l'utilisateur': {k2}", " boxplot_utilisateurs_train")
    
    freq = frequence(corpus)
    graphics.graph_repartition(freq, "mots les plus fréquents (données apprentissage)")

    b = time.time()
    print("TIME:", b-a)


    print("finished")
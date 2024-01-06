import operator
import numpy as np
import graphics
#import data_processing
import time
import scipy
import sys, os
from collections import Counter
from scipy import stats
import pandas as pd

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)

import config

path = config.paths['url']
"""
folder_to_load = sys.argv[1]
folder_to_load2 = sys.argv[2]
"""

folder_to_load = "dev"
folder_to_load2 = "train"


reviews_grades = np.load(f"{path}/processed_data/{folder_to_load}/reviews_grades.npy", allow_pickle=True).item()
reviews_users = np.load(f"{path}/processed_data/{folder_to_load}/reviews_users.npy", allow_pickle=True).item()
movie_grades = np.load(f"{path}/processed_data/{folder_to_load}/movie_grades.npy", allow_pickle=True).item()
comments = np.load(f"{path}/processed_data/{folder_to_load}/comments_clean.npy", allow_pickle=True).item()

reviews_grades2 = np.load(f"{path}/processed_data/{folder_to_load2}/reviews_grades.npy", allow_pickle=True).item()
comments2 = np.load(f"{path}/processed_data/{folder_to_load2}/comments_clean.npy", allow_pickle=True).item()

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

def get_grades_repartition(reviews_grades):
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
        if(corpus[key] is None):
            corpus[key] = ""
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
     
    # Calcul de la corrélation de Pearson
    pearson_result = scipy.stats.pearsonr(notes, lengths)
    pearson_correlation = pearson_result[0]
    #p_value = pearson_result[1]
    
    # Calcul corrélation de spearman
    res = stats.spearmanr(notes, lengths)
    spearman_correlation = res.correlation

    return pearson_correlation, spearman_correlation

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
def stats_frequent_words(reviews_grades, comments, list_notes):
    notes_by_words = {note:[] for note in list_notes}

    # Stocker tous les mots d'un commentaire associé à une note précise
    for key in comments:
        note = reviews_grades[key]
        notes_by_words[note] += comments[key]
    
    # Fréquence des mots selon la note
    freq_words = {}
    for key in notes_by_words:
        freq_words[key] = Counter(notes_by_words[key])
        freq_words[key] = dict(sorted(freq_words[key].items(), key=lambda item: item[1], reverse=True))

    return notes_by_words, freq_words

# Stat sur les notes, n est le nombre min de note d'un utilisateur pour le prendre en compte dans les stats
def stat_notes(distrib_notes, n):
    
    notes = {}
    for key, values in distrib_notes.items():
        notes[key] = []
        for value in values:
            notes[key] += [value[1]] * value[0]
    
    std = []
    mean = []
    for key, value in notes.items():
        if(len(value)>n):
            std.append(np.std(value))
            mean.append(np.mean(value))
            
    return std, mean

if __name__ == "__main__":
    ####### Generating data structures #######
    
    mean_note_by_movie = get_mean_grades_by_movie()
    distrib_notes = get_grades_repartition(reviews_grades)
    distrib_notes2 = get_grades_repartition(reviews_grades2) # Ajouter la variable grade_reviews dans la fonction
    distrib_notes_by_movie = get_grades_repartition_by_movie()
    
    grades_by_user = get_grades_by_user()
    
    distrib_notes_by_user = get_grades_repartition_by_user(grades_by_user)
    b = time.time()
   
    ######## Statistiques ########
    
    comments_char, comments_words = length_comments(comments)
    pearson_correlation_words, spearman_correlation_words = corr_quant(reviews_grades, comments_words)
    pearson_correlation_char, spearman_correlation_char = corr_quant(reviews_grades, comments_char)

    print("Longueur moyenne d'un commentaire (mots)", np.mean(list(comments_words.values())))
    print("Longueur moyenne d'un commentaire (charactères)", np.mean(list(comments_char.values())))
    
    print("Pearson coefficient correlation for comment lengths / grade (words): ", pearson_correlation_words)
    print("Pearson coefficient correlation for comment lengths / grade (char): ", pearson_correlation_char)
    
    # Stats des notes mises pour chaque utilisateurs
    std_user_0, mean_user_0 = stat_notes(distrib_notes_by_user, 0)
    print("statistiques concernant l'écart-type des notes par utilisateur:")
    print(pd.Series(std_user_0).describe())
    print("statistiques concernant la moyenne des notes par utilisateur:")
    print(pd.Series(mean_user_0).describe())
    
    std_user_5, mean_user_5 = stat_notes(distrib_notes_by_user, 5)
    print("statistiques concernant l'écart-type des notes par utilisateur ayant mis au moins 6 notes:")
    print(pd.Series(std_user_5).describe())
    print("statistiques concernant la moyenne des notes par utilisateur ayant mis au moins 6 notes:")
    print(pd.Series(mean_user_5).describe())
    
    # Stats des notes mises pour chaque films
    std_movie_0, mean_movie_0 = stat_notes(distrib_notes_by_movie, 0)
    print("statistiques concernant l'écart-type des notes par film:")
    print(pd.Series(std_movie_0).describe())
    print("statistiques concernant la moyenne des notes par film:")
    print(pd.Series(mean_movie_0).describe())

    std_movie_5, mean_movie_5 = stat_notes(distrib_notes_by_movie, 5)
    print("statistiques concernant l'écart-type des notes par film ayant au moins 6 notes")
    print(pd.Series(std_movie_5).describe())
    print("statistiques concernant la moyenne des notes par film ayant au moins 6 notes")
    print(pd.Series(mean_movie_5).describe())
    
    print("Most frequent words and their mean grade: ")
    mean_grade_by_words = most_frequent_words_and_their_mean_grade(reviews_grades, comments)
    mean_grade_by_words2 = most_frequent_words_and_their_mean_grade(reviews_grades2, comments2)
    
    i = 0
    for key in mean_grade_by_words:
        if i == 10:
            break
        
        print(mean_grade_by_words[key])
        i+= 1
    
    # Garder seulement les 10 mots les plus fréquents avec leur fréquence
    
    freq_grade_words = {key: value[2] for key, value in list(mean_grade_by_words.items())[0:10]}
    freq_grade_words2 = {key: value[2] for key, value in list(mean_grade_by_words2.items())[0:10]}

    list_notes = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    notes_by_words, freq_words = stats_frequent_words(reviews_grades, comments, list_notes)
    notes_by_words2, freq_words2 = stats_frequent_words(reviews_grades2, comments2, list_notes)
    
    print("Most frequent words for each note and their mean grade: \n ")
    for note in list_notes:
        
        i = 0
        print("note = ", note, " - Fréquence = ", distrib_notes[note])
        for key in freq_words[note].keys():
            if i == 10:
                break
        
            print(key, " - ", freq_words[note][key]/distrib_notes[note] , " - ", freq_words[note][key])
            i+= 1
        print('\n')
    
    # Voir la fréquence des mots "film" et "bien" pour chaque note
    freq_film_bien_by_note = {}
    freq_film_bien_by_note2 = {}
    freq_film_bien_by_note["film"] = {}
    freq_film_bien_by_note["bien"] = {}
    freq_film_bien_by_note2["film"] = {}
    freq_film_bien_by_note2["bien"] = {}
    
    for note in list_notes:
        freq_film_bien_by_note["film"][note] = freq_words[note]["film"]/distrib_notes[note]
        freq_film_bien_by_note["bien"][note] = freq_words[note]["bien"]/distrib_notes[note]
        freq_film_bien_by_note2["film"][note] = freq_words2[note]["film"]/distrib_notes2[note]
        freq_film_bien_by_note2["bien"][note] = freq_words2[note]["bien"]/distrib_notes2[note]
        
    ######## Generating graphs ########
    graphics.graph_repartition(distrib_notes, "Répartition des notes (" + folder_to_load + ")", "Notes")

    # Répartition des notes, comparaison entre les données de dev et d'apprentissage
    graphics.graph_note_repartition(distrib_notes, distrib_notes2, "notes", folder_to_load, folder_to_load2, "Notes") # ajouter les arguments folder_to_load et folder_to_load2 dans la fonction
    
    #k = list(movie_grades.keys())[0]
    #k2 = list(grades_by_user.keys())[0]
    #graphics.graph_boxplot(movie_grades[k], f"distribution des notes pour le film: {k}","boxplot_films_train")
    #graphics.graph_boxplot(grades_by_user [k2], f"distribution des notes pour\n l'utilisateur': {k2}", " boxplot_utilisateurs_train")

    graphics.graph_repartition_by(distrib_notes_by_movie, 4, " films (" + folder_to_load + ")")
    graphics.graph_repartition_by(distrib_notes_by_user, 4, " utilisateurs (" + folder_to_load + ")")

    graphics.graph_boxplot(distrib_notes_by_movie, 10, "films (" + folder_to_load + ")")  
    graphics.graph_boxplot(distrib_notes_by_user, 10, "utilisateurs (" + folder_to_load + ")")
    
    graphics.graph_repartition(freq_grade_words, "Fréquence des mots les plus fréquents (" + folder_to_load + ")", "Mots")
    graphics.graph_note_repartition(freq_film_bien_by_note["film"], freq_film_bien_by_note2["film"], 'Fréquence moyenne du mot film selon la note' , folder_to_load, folder_to_load2, "Mots") # ajouter les arguments folder_to_load et folder_to_load2 dans la fonction
    graphics.graph_note_repartition(freq_film_bien_by_note["bien"], freq_film_bien_by_note2["bien"], 'Fréquence moyenne du mot bien selon la note' , folder_to_load, folder_to_load2, "Mots") # ajouter les arguments folder_to_load et folder_to_load2 dans la fonction
    graphics.graph_note_repartition(freq_grade_words, freq_grade_words2, "Top 10 des mots les plus fréquents", folder_to_load, folder_to_load2, "Mots") # ajouter les arguments folder_to_load et folder_to_load2 dans la fonction


    graphics.word_cloud(mean_grade_by_words, "Wordcloud of comments (" + folder_to_load + ")")

    print("finished")

from bs4 import BeautifulSoup
import numpy as np
import re, sys, os
import requests
import tqdm

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)

import config

path = config.paths['url']
#test_movies = np.load(f"{path}/processed_data/test/reviews_movie.npy", allow_pickle=True).item()
dev_movies = np.load(f"{path}/processed_data/dev/reviews_movie.npy", allow_pickle=True).item()
train_movies = np.load(f"{path}/processed_data/train/reviews_movie.npy", allow_pickle=True).item()

def scrape_website(url):
    # Send a GET request to the URL
    try:
        response = requests.get(url)
    except:
        return ""

    data = ""
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        metadonnes = soup.find_all('div', class_="meta-body-item meta-body-info")

        regex_alphabet = re.compile('[a-zA-Z]')

        data = ""
        for donne in metadonnes:
            words = donne.text.strip().split('\n')

            for w in words:
                if regex_alphabet.search(w):
                    data += w + " "
        
    return data


def get_movies_metadata(file, movies):
    checked_movies = []
    movieid_metadata = {}

    for reviewid in tqdm.tqdm(movies):
        movieid = movies[reviewid]
        if not movieid in checked_movies:
            url = f"https://www.allocine.fr/film/fichefilm_gen_cfilm={movieid}.html"
            data = scrape_website(url)
            movieid_metadata[movieid] = data
            if not data == "":
                checked_movies.append(movieid)

    
    np.save(f"{path}/processed_data/{file}/movie_metadata.npy", movieid_metadata)
    print(movieid_metadata)
 
get_movies_metadata("dev", dev_movies)

print("finished")
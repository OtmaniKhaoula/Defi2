# Defi2

Pour générer les dictionnaires:
- Créer un dossier data et stocker les fichiers d'entraînements, de développement et de test
- Créer un dossier processed_data où l'ensemble des nouveaux fichiers seront stockés
- Exécuter le fichier data_reader.py dans le dossier tools

Pour générer les graphiques et les statistiques:
- Exécuter le fichier tools/stat.py

Pour la création du dictionnaire sentimental:
- Exécuter le fichier tools/create_dict.py

- Entraîner le modèle FastText: main.py
Les autres modèles se trouvent dans le dossier models (CNN, CamemBERT et SVM)


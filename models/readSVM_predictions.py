f = open("./SVMs/SVM-res.txt")
grades = f.readlines()
f.close()

ids_f = open("./SVMs/ids.txt")
ids = ids_f.readlines()
ids_f.close()

to_save = ""

for i in range(len(grades)):
    to_save += str(ids[i].replace('\n', '')) + " " + str(float(grades[i])/float(2)).replace('.', ',') + "\n"


target = open("../predictions/SVM-predictions.txt", "w")
target.write(to_save)
target.close()
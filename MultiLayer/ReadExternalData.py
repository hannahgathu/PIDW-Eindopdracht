import autograd.numpy as np
import csv


def ReadIrisData(file='Iris.csv'):
    """ open 'Iris.csv' """
    aantal = 150
    label = {"Iris-setosa" : 1, "Iris-versicolor" : 2, "Iris-virginica" : 3}
    x = np.zeros((aantal,4))
    y = np.zeros(aantal)

    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None) # skip headers

        #loop over rows
        for row in reader:
            k = int(row[0])-1 # index
            x[k,:] = row[1:5] # features
            y[k] = label[row[5]] # labels

    # Now, X contains 150 input samples with 4 features each:
    #(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
    # and y contains 150 labels (1, 2, 3)
    # Dit is de trainingsdata

    print("Data in '" + str(file) + "' gelezen.")
    return (x, y)



def ReadLabels(file, aantal=None):
    """ open een ubyte-bestand, en vertaal de bytes naar labels
    Invoer:
        * ubyte-bestand met labels
        * aantal labels
    Uitvoer: lijst met labels
    """
    f = open(file, 'rb')
    f.seek(8)
    y = np.fromfile(f, dtype=np.ubyte)
    y = y[:aantal]
    f.close()
    print("Labels  in '" + str(file) + "' gelezen.")
    return y


def ReadImages(file, aantal=None, h=28, b=28):
    """ open een ubyte-bestand met pixels, en vetaal de bytes
    Invoer:
        * ubyte-bestand met afbeeldingen
        * aantal afbeeldingen
        * hoogte van de afbeeldingen (in pixels)
        * breedte van de afbeeldingen (in pixels)
    Uitvoer: array met waarden 0-255 en dimensies (aantal, hoogte*breedte)
    """
    f = open(file, 'rb')
    f.seek(16)
    p = np.fromfile(f, dtype=np.ubyte)
    p = p[:aantal*h*b]
    x = np.reshape(p, [aantal, -1])
    f.close()
    print("Figuren in '" + str(file) + "' gelezen.")
    return x

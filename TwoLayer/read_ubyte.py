import autograd.numpy as np

def ReadLabels(file, aantal):
    """ open een ubyte-bestand, en vertaal de bytes naar labels
    Invoer: ubyte-bestand met labels
    Uitvoer: lijst met labels
    """
    f = open(file, 'rb')
    f.seek(8)
    y = np.fromfile(f, dtype=np.ubyte)
    y = y[:aantal]
    f.close()
    print("Labels  in '" + str(file) + "' gelezen.")
    return y

def ReadImages(file, aantal, h=28, b=28):
    """ open een ubyte-bestand met pixels, en vetaal de bytes
    Invoer: ubyte-bestand met afbeeldingen
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

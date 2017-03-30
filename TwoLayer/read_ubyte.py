import autograd.numpy as np

def ReadLabels(file):
    """ open een ubyte-bestand, en vertaal de bytes naar labels
    Invoer: ubyte-bestand met labels
    Uitvoer: lijst met labels
    """
    f = open(file, 'rb')
    f.seek(8)
    y = np.fromfile(f, dtype=np.ubyte)
    f.close()
    return y

def ReadImages(file):
    """ open een ubyte-bestand met pixels, en vetaal de bytes
    Invoer: ubyte-bestand met afbeeldingen
    Uitvoer: array met waarden 0-255 en dimensie (aantal, hoogte*breedte)
    """
    f = open(file, 'rb')
    f.seek(16)
    p = np.fromfile(f, dtype=np.ubyte)
    x = np.reshape(p, [10000, -1])
    f.close()
    return x

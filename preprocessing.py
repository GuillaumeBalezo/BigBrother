import os
import functions
import h5py
import numpy as np

def data():
    descriptors = []
    names = []
    i = 0
    for file in os.listdir("./ressources/known_peoples"):
        enc = functions.face_encoding("./ressources/known_peoples/"+file)
        if(enc != []):
            descriptors.append(enc)
            names.append(i)
            i += 1
            print(i)
    return names, descriptors

known_peoples_labels, known_peoples_encodings = data()
n = np.shape(known_peoples_labels)[0]
known_peoples_labels = np.reshape(known_peoples_labels, (n, 1))

n1, p1 = np.shape(known_peoples_labels)
n2, p2 = np.shape(known_peoples_encodings)

known_peoples = h5py.File('./known_peoples.hdf5', 'w')

known_peoples.create_dataset('known_peoples_encodings', (n2, p2), dtype=float, data=known_peoples_encodings)
known_peoples.create_dataset('known_peoples_labels', (n1, p1), dtype=int, data=known_peoples_labels)

known_peoples.flush()
known_peoples.close()

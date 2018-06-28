# IMPORTS
import functions
import cv2
import h5py
import os

# IMPORT DATASET
known_peoples = h5py.File('./known_peoples.hdf5', 'r')
known_peoples_encodings = known_peoples["known_peoples_encodings"] # (n, 128)
known_peoples_labels = []
for file in os.listdir("./ressources/known_peoples"):
    known_peoples_labels.append(str(file)[:-4]) # (n, 1)

# WEBCAM LIVE
video_capture = cv2.VideoCapture(0)

individus = -1
name = []

tmp_track = 1
nb_track = 10

while True:
    _, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    face_locations = functions.face_locations(rgb_frame, "HOG")
    tmp = len(face_locations)

    if tmp > individus or tmp_track == nb_track+1:

        # recognition
        print(' ')
        print('Phase de reconnaissance')
        print('Nombre de personnes détectées : {}'.format(len(face_locations)))
        name, coord = functions.recognition_closed(frame, face_locations, known_peoples_encodings, known_peoples_labels)
        tmp_track = 1
        #print('Nom des personnes détectées : {}'.format(name))
        #print('Coordonnées des personnes détectées : {}'.format(coord))
        print(' ')

    else:

        # tracking
        print('Phase de tracking : {} / {}'.format(tmp_track, nb_track))
        print('Nombre de personnes détectées par le HOG : {}'.format(len(face_locations)))
        face_locations = functions.face_locations(rgb_frame, "MTCNN")
        print('Nombre de personnes détectées par le MTCNN : {}'.format(len(face_locations)))
        #name, coord = functions.tracking(face_locations, frame, name, coord)
        name, coord = functions.tracking_v2(face_locations, frame, name, coord)
        tmp_track += 1
        #print('Nom des personnes détectées : {}'.format(name))
        #print('Coordonnées des personnes détectées : {}'.format(coord))

    individus = tmp

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# IMPORTS
import functions
import cv2
import h5py
import os


# CREATE DATASET
encodings = []
ids = []

# WEBCAM LIVE
video_capture = cv2.VideoCapture(0)

individus = -1
name = []

tmp_track = 0
nb_track = 10

while True:
    _, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    face_locations = functions.face_locations(rgb_frame, "HOG")
    tmp = len(face_locations)

    if tmp > individus or tmp_track == nb_track:
        # recognition
        name, coord, encodings, ids = functions.recognition_opened(frame, face_locations, encodings, ids)
        tmp_track = 0

        #print('Phase de reconnaissance')
        #print('Nombre de personnes détectées : {}'.format(len(face_locations)))
        #print('Nom des personnes détectées : {}'.format(name))
        #print('Coordonnées des personnes détectées : {}'.format(coord))
        #print('Nombre de personnes différentes reconnues depuis le début : {}'.format(len(encodings)))
        #print('Nom des personnes reconnues ; {}'.format(ids))

    else:
        # tracking
        face_locations = functions.face_locations(rgb_frame, "MTCNN")
        name, coord = functions.tracking_v2(face_locations, frame, name, coord)
        tmp_track += 1

        #print('Phase de tracking : {} / {}'.format(tmp_track, nb_track))
        #print('Nombre de personnes détectées : {}'.format(len(face_locations)))
        #print('Nom des personnes détectées : {}'.format(name))
        #print('Coordonnées des personnes détectées : {}'.format(coord))
        #print('Nombre de personnes différentes reconnues depuis le début : {}'.format(len(encodings)))
        #print('Nom des personnes reconnues ; {}'.format(ids))

    individus = tmp

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

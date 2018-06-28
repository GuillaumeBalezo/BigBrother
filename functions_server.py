## IMPORTS
import numpy as np
import dlib
import models
import cv2
from mtcnn.mtcnn import MTCNN
from pkg_resources import resource_filename
import random

## MODELS

## MODELS
face_locations_hog = dlib.get_frontal_face_detector()
face_detector_mtcnn = MTCNN()
face_landmarks_68_points = dlib.shape_predictor(resource_filename(__name__, "ressources/models/shape_predictor_68_face_landmarks.dat"))
face_encoder_resnet = dlib.face_recognition_model_v1(resource_filename(__name__, "ressources/models/dlib_face_recognition_resnet_model_v1.dat"))


## UTILS
# concatenate the name and coordinates to send
def concatenate(name,coord):
    list=[]*len(name)
    for i in range(len(name)):
        list.append([name[i],coord[i]])
    return list
# load an image
def load_image(chemin):
	# input : the path to an jpg/png image
	# output : a numpy array of the image
	image = cv2.imread(chemin)
	return np.array(image)

# rect to list
def rect_to_list(rect):
	# input : a dlib rectangle
	# output : a list of int coordinates [left, top, right, bottom]
	return [rect.left(),rect.top(), rect.right(), rect.bottom()]

# list to rect
def list_to_rect(l):
	# input : a list of int coordinates [left, top, right, bottom]
	# output : a dlib rectangle
	return dlib.rectangle(l[3], l[0], l[1], l[2])

# face distance
def face_encodings_distance(face_encoding1, face_encoding2):
	# input : two descriptors to compare (two 128-dim vectors)
	# output : the distance between those descriptors (the L1-norm)
	return np.linalg.norm(face_encoding1 - face_encoding2)

def dist(c1, c2):
	# input : 2 points ((x1, y1), (x2, y2))
	# output : the distances between them |x1 - x2| + |y1 - y2|
	left1, top1 = c1[0], c1[1]
	left2, top2 = c2[0], c2[1]
	d = np.abs(left1 - left2) + np.abs(top1 - top2)
	return d
def generate_word(longueur):
	caracteres = "azertyuiopqsdfghjklmwxcvbnAZERTYUIOPQSDFGHJKLMWXCVBN0123456789"
	mot = ""
	compteur = 0
	while compteur < longueur:
		lettre = caracteres[random.randint(0, len(caracteres)-1)]
		mot += lettre
		compteur += 1
	return mot

def retirer_doublons(coord, name, d):
	# input : coord = [(255, 123), (512, 110), (1020, 200)]
	#         name = ["Ridouane", "Maroua", "Ridouane"]
	#         d = [0.3, 0.5, 0.2]
	# output : coord = [(512, 110), (1020, 200)]
	#          name = ["Maroua", "Ridouane"]
	n = len(name)
	i = 0
	while n - i > 0:
		nom = name[i]
		sublist_d = []
		sublist_ind = []
		for j in range(n):
			if name[j] == nom:
				sublist_d.append(d[j])
				sublist_ind.append(j)
		# [0.3, 0.2], [0, 2]
		min_d = min(sublist_d)
		imin_d = sublist_d.index(min_d)
		# 0.2, 1
		for k in range(len(sublist_ind)):
			if k != imin_d:
				name = remove_index(name, sublist_ind[k])
				coord = remove_index(coord, sublist_ind[k])
		n = len(name)
		i += 1
	return name, coord

def remove_index(list, ind):
	l = []
	for i in range(len(list)):
		if i != ind:
			l.append(list[i])
	return l

## FUNCTIONS

# face locations
def face_locations(image, model="MTCNN"):
	# input : an image (numpy array)
	# output : list of face boxes coordinates (list of list of 4 elements)
	faces = []
	if model == "MTCNN":
		faces = face_detector_mtcnn.detect_faces(image)
		for i in range(len(faces)):
			faces[i] = faces[i]['box']
			faces[i][2] = faces[i][0] + faces[i][2]
			faces[i][3] = faces[i][1] + faces[i][3]
	elif model == "HOG":
		f = face_locations_hog(image)
		faces = []
		for _, face in enumerate(f):
			faces.append(rect_to_list(face))
	else:
		print("Erreur : le modèle choisi est incorrect, modèles disponibles : HOG, MTCNN")
	return faces

# face encoding
def face_encoding(chemin):
	# input : the path to an image
	# output : the descriptor of the unique face on the image
	image = load_image(chemin)
	desc = descriptors(image)

	if(len(desc) != 0):
		encoding = desc[0]
	else:
		encoding = []
		#print(chemin)

	return encoding

# descriptors
def descriptors(image):
	faces = face_locations_hog(image)
	descriptors = []
	for i, face in enumerate(faces):
		shape = face_landmarks_68_points(image, face)
		descriptor = face_encoder_resnet.compute_face_descriptor(image, shape)
		descriptors.append(np.array(descriptor))
	return descriptors


# tracking
def tracking(face_locations, frame, prev_name, prev_coord):
	coord = [(face_locations[i][0], face_locations[i][1]) for i in range(len(face_locations))]
	name = [""]*len(coord)
	for k in range(len(coord)):
		distances = [dist(coord[k], prev_coord[j]) for j in range(len(prev_coord))]
		if distances != []:
			min_distances = min(distances)
			imin_distances = distances.index(min_distances)
			name[k] = prev_name[imin_distances]
			prev_name.remove(prev_name[imin_distances])
			prev_coord.remove(prev_coord[imin_distances])
		else:
			name = ["Unknown"]*len(coord)

	j = 0
	for face_location in face_locations:
		left, top, right, bottom = face_location[0], face_location[1], face_location[2], face_location[3]
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name[j], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
		j += 1

	return name, coord

def tracking_v2(face_locations, frame, prev_name, prev_coord):
	#print('Noms précédents : {}'.format(prev_name))
	#print('Coordonnées précédentes : {}'.format(prev_coord))
	coord = [(face_locations[i][0], face_locations[i][1]) for i in range(len(face_locations))]
	name = [""]*len(coord)
	d = [0]*len(coord)
	for k in range(len(coord)):
		distances = [dist(coord[k], prev_coord[j]) for j in range(len(prev_coord))]
		if distances != []:
			min_distances = min(distances)
			imin_distances = distances.index(min_distances)
			name[k] = prev_name[imin_distances]
			d[k] = min_distances
		else:
			name[k] = "Unknown"
	#print('Nouveaux noms ; {}'.format(name))
	#print('Nouvelles coordonnées ; {}'.format(coord))
	#print(d)
	name, coord = retirer_doublons(coord, name, d)
	#print('Nouveaux noms ; {}'.format(name))
	#print('Nouvelles coordonnées ; {}'.format(coord))
	for j in range(len(coord)):
		left, top, right, bottom = coord[j][0], coord[j][1], face_locations[j][2], face_locations[j][3]
		#print(left, top, right, bottom)
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
		font = cv2.FONT_HERSHEY_DUPLEX
		#<sprint(name[j])
		cv2.putText(frame, name[j], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

	return name, coord

# recognition

def recognition_opened(frame, face_locations, encodings, ids):
	rgb_frame = frame[:, :, ::-1]
	face_encodings = descriptors(rgb_frame)
	name = ["Unknown" for i in range(len(face_locations))]
	coord = [(face_locations[i][0], face_locations[i][1]) for i in range(len(face_locations))]

	j = 0
	for face_location, face_encoding in zip(face_locations, face_encodings):
		if(len(encodings) == 0):
			encodings.append(face_encoding)
			ids.append(generate_word(5))
			name[j] = ids[0]
		else:
			distances = [face_encodings_distance(face_encoding, tmp) for tmp in encodings]
			#print(distances)
			if distances != []:
				min_distances = min(distances)
				imin_distances = distances.index(min_distances)
				if min_distances > 0.60:
					encodings.append(face_encoding)
					ids.append(generate_word(5))
					name[j] = ids[len(ids) - 1]
				else:
					name[j] = ids[imin_distances]

		left, top, right, bottom = face_location[0], face_location[1], face_location[2], face_location[3]
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name[j], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
		j += 1

	return name, coord, encodings, ids

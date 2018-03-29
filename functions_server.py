## IMPORTS
import numpy as np
import dlib
import models
import imageio

## MODELS

# face locations
face_locations_hog = dlib.get_frontal_face_detector()

# face landmarks
face_landmarks_68_points = dlib.shape_predictor(models.shape_predictor_68_face_landmarks())

# face encodings
face_encoder_resnet = dlib.face_recognition_model_v1(models.dlib_face_recognition_resnet_model_v1())


## UTILS
# concatenate the name and coordinates to send
def concatenate(name,coord):
    list=[[]]*len(name)
    for i in range(len(name)):
        list[i].append([name[i]])
        list[i].append(coord[i])
    return list
def prepare_to_send(name,coord):
    str_to_send=str(len(name))
    for i in range(len(name)):
        str_to_send+=str(len(name[i]))+name[i]
    for i in range(len(coord)):
        str_to_send+=str(len(str(coord[i][0])))+str(coord[i][0])+str(len(str(coord[i][1])))+str(coord[i][1])
    return bytes(str_to_send, 'utf-8')
# load an image
def load_image(chemin):
    # input : an jpg/png/... image
    # output : a numpy array
    image = imageio.imread(chemin)
    return np.array(image)

# rect to list
def rect_to_list(rect):
    return [rect.left(), rect.top(), rect.right(), rect.bottom()]

# list to rect
def list_to_rect(l):
    return dlib.rectangle(l[3], l[0], l[1], l[2])

# face distance
def face_encodings_distance(face_encoding1, face_encoding2):
    # input : two descriptors to compare
    # output : the distance between those descriptors
    return np.linalg.norm(face_encoding1 - face_encoding2)

def dist(c1, c2):
    left1, top1 = c1[0], c1[1]
    left2, top2 = c2[0], c2[1]

    d = np.abs(left1 - left2) + np.abs(top1 - top2)

    return d

def ind_min(l):
    imin = 0
    min = l[0]
    for j in range(1, len(l)):
        if l[j] < min:
            min = l[j]
            imin = j
    return imin

## FUNCTIONS

# face locations
def face_locations(image):
    # input : an image (jpg or numpy?)
    # output : a list of face locations of the faces in the image
    faces = face_locations_hog(image)
    f = []
    for i, face in enumerate(faces):
        f.append(rect_to_list(face))
    return f

# face encoding
def face_encoding(chemin):
    # input : the path to an image
    # output : the descriptor of the unique face on the image
	image = load_image(chemin)
	encoding = descriptors(image)[0]
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
    name = [" "]*len(coord)
    for k in range(len(coord)):
        distances = [dist(coord[k], prev_coord[j]) for j in range(len(prev_coord))]
        if distances != []:
            imin = ind_min(distances)
            name[k] = prev_name[imin]
        else:
            name = ["Unknown"]*len(coord)
    return name, coord

# recognition
def recognition(frame, face_locations, known_peoples_encodings, known_peoples_labels):
    # input : a frame, face locations, ref dataset
    # output : names+coords, customized frame
    face_encodings = descriptors(frame)
    name = ["Unknown" for i in range(len(face_locations))]
    coord = [(face_locations[i][0], face_locations[i][1]) for i in range(len(face_locations))]

    j = 0
    for face_location, face_encoding in zip(face_locations, face_encodings):
        distances = [face_encodings_distance(encoding, face_encoding) for encoding in known_peoples_encodings]
        n = len(distances)
        if distances != []:
            imin = ind_min(distances)
        name[j] = known_peoples_labels[imin]
        #for i in range(n):
        #    if distances[i] < 0.55:
        #        name[j] = known_peoples_labels[i]
        j += 1
    return name, coord

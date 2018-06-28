from pkg_resources import resource_filename

# face detector
def dlib_cnn_face_detection_model_v1():
    return resource_filename(__name__, "ressources/models/mmod_human_face_detector.dat")

# face landmarks
def shape_predictor_68_face_landmarks():
    return resource_filename(__name__, "ressources/models/shape_predictor_68_face_landmarks.dat")

# face encodings
def dlib_face_recognition_resnet_model_v1():
    return resource_filename(__name__, "ressources/models/dlib_face_recognition_resnet_model_v1.dat")

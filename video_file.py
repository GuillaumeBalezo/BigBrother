# IMPORTS
import functions
import cv2
import h5py
import os

# IMPORT DATASET
known_peoples = h5py.File('./known_peoples.hdf5', 'r')
known_peoples_encodings = known_peoples["known_peoples_encodings"]
known_peoples_labels = []
for file in os.listdir("./ressources/known_peoples"):
    known_peoples_labels.append(str(file)[:-4])

# IMPORT INPUT VIDEO FILE
input_title = 'a_reid.mp4'
x, y = 1920, 1080 # resolution
frame_rate = 30.0 # fps
temps = 49
input_video = cv2.VideoCapture(input_title)

# CREATE OUTPUT VIDEO FILE
output_title = 'reid.mp4'
output_video = cv2.VideoWriter(output_title, cv2.VideoWriter_fourcc(*'MP4V'), frame_rate, (x, y))

# PROCESSING...
i = 0
N = frame_rate*temps

individus = -1
name = []
if(input_video.isOpened() == False):

    print("Error opening video stream or file")

else:

    while(input_video.isOpened()):
        ret, frame = input_video.read()
        rgb_frame = frame[:, :, ::-1]

        face_locations = functions.face_locations_mtcnn(rgb_frame)
        tmp = len(face_locations)

        if tmp > individus:
            # recognition
            name, coord = functions.recognition(frame, face_locations, known_peoples_encodings, known_peoples_labels)
            print("Phase de reconnaissance")
        else:
            # tracking
            name, coord = functions.tracking(face_locations, frame, name, coord)
            print("Phase de tracking")

        individus = tmp

        output_video.write(frame)

        print(str(i)+'/'+str(N))
        i += 1

    input_video.release()
    cv2.destroyAllWindows()

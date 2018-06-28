## UTILS
#Create the windows in the frame
import cv2
def create_frame(data,frame):
    for i in range(len(data)):
        left, top, right, bottom = data[i][1][0],data[i][1][1],data[i][1][2],data[i][1][3]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, data[i][0], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

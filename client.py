import cv2
import socket
import base64
import numpy as np
import pickle
import functions_client

IP_SERVER = "18.184.201.224"
PORT_SERVER = 8089
TIMEOUT_SOCKET = 10
SIZE_PACKAGE = 4096
DEVICE_NUMBER = 0


if __name__ == '__main__':
    cap = cv2.VideoCapture(DEVICE_NUMBER)
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connection.settimeout(TIMEOUT_SOCKET)
    connection.connect((IP_SERVER, PORT_SERVER))

    while True:
        try:
            ret, frame = cap.read()
            a = b'\r\n'
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            ret, frame_compress = cv2.imencode('.jpg', frame, encode_param)
            data = frame_compress.tostring()
            da = base64.b64encode(data)
            connection.sendall(da + a)
            data=connection.recv(SIZE_PACKAGE)
            data=pickle.loads(data)
            functions_client.create_frame(data,frame)
            cv2.namedWindow("Video",cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Video",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print("[Error] " + str(e))

    connection.close()
    cap.release()

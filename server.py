import time
import socket
import numpy as np
import base64
from threading import Thread
import functions_server as functions
import preprocessing
import h5py
import os
import pickle

SERVER_IP = ""
SERVER_PORT = 8089
MAX_NUM_CONNECTIONS = 20

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
COLOR_PIXEL = 3  # RGB

class ConnectionPool(Thread):

    def __init__(self, ip_, port_, conn_,known_peoples_encodings_,known_peoples_labels_):
        Thread.__init__(self)
        self.ip = ip_
        self.port = port_
        self.conn = conn_
        self.known_peoples_encodings=known_peoples_encodings_
        self.known_peoples_labels=known_peoples_labels_
        print("[+] New server socket thread started for " + self.ip + ":" +str(self.port))

    def run(self):
        try:
            individus = -1
            name = []
            while True:
                fileDescriptor = conn.makefile(mode='rb')
                result = fileDescriptor.readline()
                fileDescriptor.close()
                result = base64.b64decode(result)
                frame = np.fromstring(result, dtype=np.uint8)
                frame_matrix = np.array(frame)
                frame_matrix = np.reshape(frame_matrix, (IMAGE_HEIGHT, IMAGE_WIDTH,COLOR_PIXEL))
                face_locations = functions.face_locations(frame_matrix)
                tmp = len(face_locations)
                if tmp > individus:
                    # recognition
                    name, coord = functions.recognition(frame_matrix, face_locations, self.known_peoples_encodings, self.known_peoples_labels)   
                    #print("Phase de reconnaissance")
                else:
                    # tracking
                    name, coord = functions.tracking(face_locations, frame_matrix, name, coord)
                    #print("Phase de tracking")
                individus = tmp
                data=functions.concatenate(name,face_locations)
                data=pickle.dumps(data)
                conn.sendall(data)
        except Exception as e:
            print("Connection lost with " + self.ip + ":" + str(self.port) +"\r\n[Error] " + str(e.message))
        self.conn.close()

if __name__ == '__main__':
    known_peoples = h5py.File('./known_peoples.hdf5', 'r')
    known_peoples_encodings = known_peoples["known_peoples_encodings"] # (n, 128)
    known_peoples_labels = []
    for file in os.listdir("./ressources/known_peoples"):
        known_peoples_labels.append(str(file)[:-4]) # (n, 1)
    print("Waiting connections...")
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connection.bind((SERVER_IP, SERVER_PORT))
    connection.listen(MAX_NUM_CONNECTIONS)
    while True:
        (conn, (ip, port)) = connection.accept()
        thread = ConnectionPool(ip, port, conn,known_peoples_encodings,known_peoples_labels)
        thread.start()
    connection.close()

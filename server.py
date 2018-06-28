import time
import socket
import numpy as np
import base64
from threading import Thread
import functions_server as functions
import h5py
import os
import pickle
import io
from PIL import Image

SERVER_IP = ""
SERVER_PORT = 8089
MAX_NUM_CONNECTIONS = 20

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
            tmp_track = 0
            nb_track = 10
            while True:
                connection_thread=self.conn
                fileDescriptor = connection_thread.makefile(mode='rb')
                result_temp = fileDescriptor.readline()
                fileDescriptor.close()
                result = base64.b64decode(result_temp)
                im = Image.open(io.BytesIO(result))
                width, height = im.size
                frame_matrix = np.array(im)
                face_locations = functions.face_locations(frame_matrix, "HOG")
                tmp = len(face_locations)
                if tmp > individus or tmp_track == nb_track+1:
                    # recognition
                    name, coord = functions.recognition_closed(frame_matrix, face_locations, self.known_peoples_encodings, self.known_peoples_labels)
                    tmp_track = 1
                    #print("Phase de reconnaissance")
                else:
                    # tracking
                    face_locations = functions.face_locations(frame_matrix, "MTCNN")
                    name, coord = functions.tracking_v2(face_locations, frame_matrix, name, coord)
                    tmp_track +=1
                    #print("Phase de tracking")
                individus = tmp
                data=functions.concatenate(name,face_locations)
                data=pickle.dumps(data,2) #python2 sinon pas de chiffre pour python3
                connection_thread.sendall(data)

                #Test pour Android
                # data=pickle.dumps([['Test', [194, 154, 373, 333]]],2)
                # connection_thread.sendall(data)

        except Exception as e:
            print("Connection lost with " + self.ip + ":" + str(self.port) +"\r\n[Error] " + str(e))#e.message
        self.conn.close()

if __name__ == '__main__':
    known_peoples = h5py.File('./known_peoples.hdf5', 'r')
    known_peoples_encodings = known_peoples["known_peoples_encodings"] # (n, 128)
    known_peoples_labels = []
    for file in os.listdir("./ressources/known_peoples"):
        known_peoples_labels.append(str(file)[:-4]) # (n, 1)
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connection.bind((SERVER_IP, SERVER_PORT))
    connection.listen(MAX_NUM_CONNECTIONS)
    print("Waiting connections on "+SERVER_IP+":"+str(SERVER_PORT)+"...")
    while True:
        (conn_, (ip, port)) = connection.accept()
        thread = ConnectionPool(ip, port, conn_,known_peoples_encodings,known_peoples_labels)
        thread.start()
    connection.close()

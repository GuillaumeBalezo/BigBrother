# BigBrother

 Face detection and re-identifaction of people from a videostream 

BigBrother, why a name like that? \
At first, we were 9 students that didn't know anything about computer vision, cloud neither deeplearning. This project was a good start for us to learn such interesting fields.\
We wanted to show what only a couple of students can achieve only in less than a year and by starting with a beginner level.\
In our algorithm we use:
- mtcnn network (to detect people)
- facenet (to identify people)
- Hog (to detect quickly people during the tracking)

------
There are two different versions of our algorithm: a closed base and a opened base version.
- In the closed base, you have to manually put one picture of each persons you want to identify. The pictures should have the following name: Name_FirstName.jpg (other usual format should work) and should be in the folder /ressource/know_peoples. You need to process preprocessing.py each time you modify the database.
- In the opened base, the algorithm automatically associates a random identifier on each person it detects for the first time, so that if the same person is detected later it will have the same id (the identifiers are deleted each time you stop it)

### Branches content

- master:\
You can run the algorithm locally on your computer.

- server_mtcnn_openedbase:\
On this branch, you can find the opened-based version of our algorithm that you can deploy on a computer with performing GPUs (on AWS for instance).\
You will use the client version (that you can find on the client branch), in order to connect and send the video stream coming from your webcam to the server.

- server_mtcnn_closedbase:\
As before, it's the closed-base version, you can used to deploy the algorithm on a server.

- client:\
Use this version to send your video stream coming from your webcam to the server. You just have to specify the IP adress and the port of the server.

### Installs
- In each branch, you will find a requirements.txt, you just have to launch the following command from a shell (ubuntu here):
sudo pip install -r requirements.txt
sudo pip3 install -r requirements.txt
- On the master branch, you have to process some other installs:
     1. OpenCv: You can install it with pip (pip install opencv-python) but you can have some errors because it's a quick install, so if it doesn't work try ti follow this tutorial:\
     https://docs.opencv.org/3.4.3/d2/de6/tutorial_py_setup_in_ubuntu.html  \
     If you use the server version don't install OpenCv and skip this step.
     2. (Optional) If you want to speed the run time of the algorithm: Install Cuda, CuDNN (check what version you should use according to your GPU) and "install tensorflow-gpu" with pip (Once again be carreful of the version you will use according to the version of Cuda and CuDNN)\
     https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html For Cuda Toolkit v.10\
     If you use the client version skip this step. It would be relevant to process this step, if you have on a computer or a server with a powerfull GPU.\
     For more information check "Install_dependencies_on_GPU.md"

##How to use it
- Locally: if you want to use the closedbase version:
   1. 
   2.
-----
### Results
<iframe width="560" height="315" src="https://www.youtube.com/embed/P8l9K7zncbE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
-----
### Our team
![alt text](https://raw.githubusercontent.com/GuillaumeBalezo/BigBrother/master/ressources/unknown_peoples/image1.jpg)

### Collaboration
This project had been achieved collaboratively with the start-up Watiz and Telecom SudParis thanks to this educative project GATE.

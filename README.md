# Predicting-Student-Attentiveness-using-OpenCV
The model will predict if a student is attentive or not through facial parameter received through the student's webcam using Face Detection, Drowsiness Detection, and Head Pose Estimation

# Libraries Used (To be installed to execute program)
1. [cv2](https://pypi.org/project/opencv-python/)
2. [dlib](http://dlib.net/)
3. [numpy](https://numpy.org/install/)
4. [mediapipe](https://google.github.io/mediapipe/getting_started/install.html)
5. [imutils](https://pypi.org/project/imutils/#files)
6. [scipy](https://scipy.org/install/)

# Information :
1. [data](data) : The data folder contains test images for cam_calibration.py and a '.dat' file used for placing landmarks on the face using dlib library.
2. [face detectors](face%20detectors) : This folder contains two face detector models that were used to implement Drowsiness Detection and Head Pose Estimation.
3. [helpermod](helpermod) : This folder contains two helper modules used for implementing Head Pose Estimation.

4. [Drowsiness_Detection.py](Drowsiness_Detection.py) : The main program for implementing Drowsiness Detection to predict Student Attentiveness.
5. [cam_calibration.py](cam_calibration.py) : This program is to be run before running headposedlib.py to implement Head Pose Estimation correctly.
6. [headposedlib.py](headposedlib.py) : This contains the main program for implementing Head Pose Estimation to predict Student Attentiveness.

# Steps to run the Program :
1. Install all the required library packages in your python environment using "pip install"
2. Drowsiness_Detection.py can then be run. A new window will popup with your webcamera turning on. To exit the program , press the ESC key.
3. Run cam_calibration.py, this will give the approximate focal length as output. This focal length has to be modified in the headposedlib.py for correct implementation.
4. In headposedlib.py, modify the focal length 'f' to the nearest intger value received as output of the cam_calibration.py program.

![image](https://user-images.githubusercontent.com/78135669/145556212-e2795a21-58cd-4875-a7ac-1782cf889519.png)

5. After updating the focal length 'f', run headposedlib.py. A new window will popup with your webcamera turning on and your video will displayed on this window.To exit the program, press the ESC key.

##### Note : The face detector models used to run Drowsiness_Detection.py and headposedlib.py are given for reference.
Both these programs can be run after the required libraries are installed. Both these programs will open a new window with your video capture through the webcam and will detect the face on the video. To exit these programs, press the ESC key. 

## Contributors
Johann Kyle Pinto
Reg N0. : 20BKT0009

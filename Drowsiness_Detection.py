from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

#Threshold value which suggests closed eyes	
thresh = 0.27 
#Checking for some n frames
frame_check = 20
#Detect face
detect = dlib.get_frontal_face_detector()
# Dat file is the crux of the code
predict = dlib.shape_predictor(
	"data\shape_predictor_68_face_landmarks.dat")

#Getting the start and end points for both eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap=cv2.VideoCapture(0)
flag=0
attn=1
while True:
	GAZE = 'Face not detected'

	ret, img=cap.read()
	img = imutils.resize(img, width=450)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	txt='Not Attentive'
	for subject in subjects:
		structure = predict(gray, subject)
		#converting to NumPy Array 
		structure = face_utils.shape_to_np(structure)
		#Draw rectangle for face detection
		if subjects != []:
			for subject in subjects:
				x = subject.left()
				y = subject.top()
				w = subject.right() - x
				h = subject.bottom() - y
				cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
				txt = 'Attentive'
				GAZE = ''

		leftEye = structure[lStart:lEnd]
		rightEye = structure[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		#Bordering eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < thresh:
			flag += 1
			if flag >= frame_check:
				cv2.putText(img, "********DROWSINESS DETECTED!**********", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				# print ("Drowsy")
				txt='Not Attentive'
		else:
			flag = 0
	cv2.putText(img, GAZE, (10, 30),
             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.putText(img, txt, (10, 325),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.imshow("Frame",img)
	#ESC key to exit
	key = cv2.waitKey(10) & 0xFF
	if key == 27:
		break
cv2.destroyAllWindows()
cap.release() 

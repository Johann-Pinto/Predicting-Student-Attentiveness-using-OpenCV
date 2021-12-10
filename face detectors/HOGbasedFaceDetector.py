import dlib
import cv2


detect = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)
while True:
	GAZE = 'Face not detected'
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)

	for subject in subjects:
		x = subject.left()
		y = subject.top()
		w = subject.right() - x
		h = subject.bottom() - y
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		GAZE = ''
	cv2.putText(img, GAZE, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
	cv2.imshow("Frame", img)
	key = cv2.waitKey(1) & 0xFF
	#ESC key to exit
	if key == 27:
		break
cv2.destroyAllWindows()
cap.release()

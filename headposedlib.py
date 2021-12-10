import cv2
import dlib
import numpy as np

#import Face Recognition libraries
import mediapipe as mp

# helper modules
from helpermod.drawFace import draw
import helpermod.reference_world as world

# **********************************************************
# SET THE FOCAL LENGTH OF YOUR CAMERA
# Round it off to an integer value

f = 534

# **********************************************************

#Settingup MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5)

PREDICTOR_PATH = "data\shape_predictor_68_face_landmarks.dat"

face3Dmodel = world.ref3DModel()


def main():
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    cap = cv2.VideoCapture(0)

    while True:
        GAZE = "Face Not Detected"
        txt = 'Not Attentive'
        ret, img = cap.read()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                location = detection.location_data
                relative_bounding_box = location.relative_bounding_box
                x_min = relative_bounding_box.xmin
                y_min = relative_bounding_box.ymin
                widthh = relative_bounding_box.width
                heightt = relative_bounding_box.height

                try:
                    absx, absy = mp_drawing._normalized_to_pixel_coordinates(
                    x_min, y_min, w, h)
                    abswidth, absheight = mp_drawing._normalized_to_pixel_coordinates(
                    x_min+widthh, y_min+heightt, w, h)

                except:
                    cv2.putText(image, 'Please Move Towards Camera', (20, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            newrect = dlib.rectangle(absx, absy, abswidth, absheight)
            cv2.rectangle(image, (absx, absy), (abswidth, absheight),
                          (0, 255, 0), 2)
            shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), newrect)

            draw(image, shape)

            refImgPts = world.ref2dImagePoints(shape)

            height, width, channels = img.shape
            # focalLength = args["focal"] * width
            focalLength = f * width
            cameraMatrix = world.cameraMatrix(
                focalLength, (height / 2, width / 2))

            mdists = np.zeros((4, 1), dtype=np.float64)

            # calculate rotation and translation vector using solvePnP
            success, rotationVector, translationVector = cv2.solvePnP(
                face3Dmodel, refImgPts, cameraMatrix, mdists)

            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

            #  draw nose line
            p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
            p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
            cv2.line(image, p1, p2, (110, 220, 0),
                     thickness=2, lineType=cv2.LINE_AA)

            # calculating euler angles
            rmat, jac = cv2.Rodrigues(rotationVector) #CONVERT ROTATION MATRIX TO ROTATION VECTOR 
            #jac is the jacobian matrix....not required
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # angles[1] <-20 indicates student is looking left
            # angles[1] > 20 indicates student is looking right
            if angles[1] < -20 or angles[1] > 20 :
                GAZE = "Kindly look towards the screen"
                txt = 'Not Attentive'

            else:
                # GAZE = "Forward"
                GAZE = ""
                txt = 'Attentive'

        cv2.putText(image, GAZE, (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        cv2.putText(image, txt, (10, 475),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Head Pose", image)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # path to your video file or camera serial
    main()

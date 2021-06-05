# This module will detect the eye closure of the user

# import the necessary libraries
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2
import numpy as np

# define two constants
# 1.    a Threshold Value, to compare the eye aspect ratio with,
#       which will indicate a blink
# 2.    the number of consecutive frames the eye must
#       be below the threshold value which would set
#       off the alert message.
EYE_THRESHOLD = 0.3
EYE_FRAMES = 48

# initialize the frame counter as well as a boolean
# used to indicate if the alarm is going off
# and the PATH for the landmarks
COUNTER = 0
ALARM_ON = False
PATH = "data/shape_predictor_68_face_landmarks.dat"


# method to display the ALERT
def sound_alarm():
    print("ALERT: Eyes Closed!")


def compute_eye_aspect_ratio(eye):

    #         [1]*   [2]*
    #    [0]*     eye     [3]*
    #         [5]*   [4]*

    # compute the euclidean distances from scipy library
    # between the two sets of vertical eye landmarks (x, y) coordinates
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    c = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    eye_ratio = (a + b) / (2.0 * c)

    # return the eye aspect ratio
    return eye_ratio


def tester():

    global COUNTER, ALARM_ON
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PATH)
    # grab the indexes of the facial landmarks for the
    # left and right eye, respectively
    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    look_up_table = np.empty((1, 256), np.uint8)
    gamma = 0.5

    for i in range(256):
        look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.LUT(frame, look_up_table)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        if len(rects) > 0:
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y) coordinates to a NumPy array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                left_eye = shape[l_start:l_end]
                right_eye = shape[r_start:r_end]
                left_eye_aspect_ratio = compute_eye_aspect_ratio(left_eye)
                right_eye_aspect_ratio = compute_eye_aspect_ratio(right_eye)

                # calculate the average of eye aspect ratios
                eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

                # visualization
                # compute the convex hull for the left and right eye

                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if eye_aspect_ratio <= EYE_THRESHOLD:
                    COUNTER += 1

                    # sound the alarm if eyes were closed for a certain
                    # amount of time
                    if COUNTER >= EYE_FRAMES:
                        if not ALARM_ON:
                            ALARM_ON = True
                            sound_alarm()

                        # draw alert text on the frame
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)

                # otherwise, the eye aspect ratio is not below the blink
                # threshold, so reset the counter and alarm
                else:
                    COUNTER = 0
                    ALARM_ON = False

                # draw the computed eye aspect ratio on the frame to help
                # with debugging and setting the correct eye aspect ratio
                # thresholds and frame counters
                cv2.putText(frame,
                            "EAR: {:.2f}".format(eye_aspect_ratio),
                            (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # cleanup
    cv2.destroyAllWindows()
    del cap


tester()

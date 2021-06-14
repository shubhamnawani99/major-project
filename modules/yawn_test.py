import cv2
import face_alignment
from yawn import is_yawning
import time

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


# show output
def show_output(frame):
    cv2.imshow('Yawn test', frame)
    cv2.waitKey(1)


def get_face_landmarks(frame):
    return fa.get_landmarks(frame)


# helper method to get FPS (Frames Per Second)
def get_fps(frame, prev_frame_time):

    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()

    # Calculating the fps
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    # putting the FPS count on the frame
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    return prev_frame_time


# tester method
def test():
    cap = cv2.VideoCapture(0)
    prev_frame_time = 0
    while cap.isOpened():

        ret, frame = cap.read()

        frame = cv2.resize(frame, (480, 360))  # Resize image

        prev_frame_time = get_fps(frame, prev_frame_time)
        face_rect = get_face_landmarks(frame)

        if face_rect is None:
            print("No Face Found!")
            show_output(frame)
            continue

        face_keypoints = face_rect[0]
        yawn = is_yawning(face_keypoints)

        if yawn:
            print("The participant is yawning!")

        mouth = face_keypoints[48:60]
        cv2.circle(frame, (int(mouth[0][0]), int(mouth[0][1])), 3, (255, 0, 0), -1)
        cv2.circle(frame, (int(mouth[6][0]), int(mouth[6][1])), 3, (255, 0, 0), -1)
        cv2.circle(frame, (int(mouth[3][0]), int(mouth[3][1])), 3, (0, 0, 255), -1)
        cv2.circle(frame, (int(mouth[9][0]), int(mouth[9][1])), 3, (0, 0, 255), -1)
        show_output(frame)

    cap.release()
    cv2.destroyAllWindows()


test()

import cv2
import face_alignment
from yawn import is_yawning_3d


# show output
def show_output(frame):
    cv2.imshow('Yawn test', frame)
    cv2.waitKey(1)

#
# def get_face_landmarks_2d(frame):
#     fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
#     return fa.get_landmarks(frame)
#

def get_face_landmarks_3d(frame):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
    return fa.get_landmarks(frame)


# tester method
def test():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():

        ret, frame = cap.read()
        # face_rect = get_face_landmarks_2d(frame)
        face_rect_3d = get_face_landmarks_3d(frame)

        if face_rect_3d is None:
            print("No Face Found!")
            show_output(frame)
            continue

        face_keypoints = face_rect_3d[0]

        yawn = is_yawning_3d(face_keypoints)

        if yawn:
            print("The participant is yawning!")

        mouth = face_keypoints[48:60]
        cv2.circle(frame, (int(mouth[0][0]), int(mouth[0][1])), 3, (255, 0, 0), -1)
        cv2.circle(frame, (int(mouth[6][0]), int(mouth[6][1])), 3, (255, 0, 0), -1)
        cv2.circle(frame, (int(mouth[3][0]), int(mouth[3][1])), 3, (0, 0, 255), -1)
        cv2.circle(frame, (int(mouth[9][0]), int(mouth[9][1])), 3, (0, 0, 255), -1)

        show_output(frame)


test()

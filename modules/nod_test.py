import cv2
import face_alignment
from landmarks3d import get_face_keypoints, calculate_vector
from nod_utils import is_nodding


# show output
def show_output(frame):
    cv2.imshow('Nodding test', frame)
    cv2.waitKey(1)


# tester method
def test():
    cap = cv2.VideoCapture(0)
    prev_frame_time = 0

    nod_list = []
    idx = 0
    while cap.isOpened():

        ret, frame = cap.read()

        # frame = cv2.resize(frame, (480, 360))  # Resize image

        # prev_frame_time = get_fps(frame, prev_frame_time)
        face_rect = get_face_keypoints(frame)

        if face_rect is None:
            print("No Face Found!")
            show_output(frame)
            continue

        face_keypoints = face_rect[0]
        cur_vector = calculate_vector(face_keypoints)

        nod = False

        if idx < 7:
            nod_list.append(cur_vector)

        else:
            nod_list[idx % 7] = cur_vector
            nod = is_nodding(nod_list)

        if nod:
            font = cv2.FONT_HERSHEY_SIMPLEX
            message = "Nodding Detected!"
            cv2.putText(frame, message, (7, 450), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

        nose = face_keypoints[31:35]
        cv2.circle(frame, (int(nose[0][0]), int(nose[0][1])), 3, (255, 0, 0), -1)
        # cv2.circle(frame, (int(nose[1][0]), int(nose[1][1])), 3, (255, 0, 0), -1)
        # cv2.circle(frame, (int(nose[2][0]), int(nose[2][1])), 3, (255, 0, 0), -1)
        cv2.circle(frame, (int(nose[3][0]), int(nose[3][1])), 3, (255, 0, 0), -1)
        show_output(frame)

        idx += 1

    cap.release()
    cv2.destroyAllWindows()


test()


import cv2


cap = cv2.VideoCapture("../resources/college_test_call_final.mp4")


# main thread
while True:

    # read from the video capture
    ret, frame = cap.read()

    # break if frame not available
    if not ret:
        print('=============END==================')
        break

    # show the video frame
    show_frame = cv2.resize(frame, (960, 540))  # Resize image
    cv2.imshow('Attention span detection demo', show_frame)
    k = cv2.waitKey(50)
    if k == ord("q"):
        exit(1)

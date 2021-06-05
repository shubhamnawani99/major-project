# import the necessary libraries
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# open a capture window using the VideoCapture() method
cap = cv2.VideoCapture(0)

# run loop till the capture window is opened
while cap.isOpened():

    # get the next frame from the window and its status in ret
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # draw a rectangle around the detected eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_frame, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 2)

    cv2.imshow('frames', frame)

    # escape key is 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

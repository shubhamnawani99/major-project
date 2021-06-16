from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from itertools import repeat
from queue import Queue
from threading import Thread

import cv2
import matplotlib.pyplot as plt

from buffer_utils import Buffer
from chopper_utils import chop
from classifier_utils import classify
from executor_utils import process_and_upload


def process_frame():
    # chop the frame into different parts
    # to get the users from each frame
    images = chop(frame)

    # print current participation strength
    print('The participation strength is :', len(images))

    # reset the buffer (the set of current people frame)
    buffer.reset_people()

    # upload the images and process them
    with ThreadPoolExecutor() as master:
        master.map(process_and_upload, images, repeat(buffer))

    # perform the classification
    classes, attentions = classify(buffer)

    # put the classification result in a queue
    return_values.put((classes, attentions))

    # set the presence of a person in the frame
    # if person not in the current frame,
    # assign false in the presence dict
    buffer.set_presences()


# define the frame, buffer and return values
buffer = Buffer()
frame = None
return_values = Queue()

cap = cv2.VideoCapture("../resources/college_test_call_final.mp4")

# define the main processing thread
processing_thread = Thread(target=process_frame)

attentions = defaultdict(lambda: [])

# main thread
while True:

    # read from the video capture
    ret, frame = cap.read()

    # break if frame not available
    if not ret:
        break

    # show the video frame
    show_frame = cv2.resize(frame, (960, 540))  # Resize image
    cv2.imshow('Attention span detection demo', show_frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        exit(1)

    # if the main thread is not alive
    # initial process and start
    if not processing_thread.is_alive():

        # while we get the return values
        while not return_values.empty():
            classes, scores = return_values.get()
            for key in scores:
                attentions[key].append(scores[key])

        # set the main thread
        processing_thread = Thread(target=process_frame)

        # run the main processing thread
        processing_thread.start()

from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from itertools import repeat
from queue import Queue
from threading import Thread

import cv2
import json

from buffer_utils import Buffer
from chopper_utils import chop
from classifier_utils import classify
from executor_utils import process_and_upload


def run():
    def process_frame():
        # chop the frame into different parts
        # to get the users from each frame
        images = chop(frame)

        # reset the buffer (the set of current people frame)
        buffer.reset_people()

        # upload the images and process them
        with ThreadPoolExecutor() as master:
            master.map(process_and_upload, images, repeat(buffer))

        # perform the classification
        class_predictions, scores = classify(buffer)

        # put the classification result in a queue
        return_values.put(class_predictions)

        # set the presence of a person in the frame
        # if person not in the current frame,
        # assign false in the presence dict
        buffer.set_presences()
        print()

    def save_file_to_json(attentions_dict):
        # the json file where the output must be stored
        out_file = open("attentions.json", "w")
        json.dump(attentions_dict, out_file, indent=6)
        out_file.close()

    def save_file(class_values):
        classes = class_values.get()
        attention_list = list()
        for key in classes:
            attentions = defaultdict()
            attentions["name"] = key
            attentions["attention"] = classes[key]
            attention_list.append(attentions)
        save_file_to_json(attention_list)

    # define the frame, buffer and return values
    buffer = Buffer()
    frame = None
    return_values = Queue()

    # cap = cv2.VideoCapture("../resources/dark_subs720.mp4")
    cap = cv2.VideoCapture("../resources/TestCall.mp4")

    # define the main processing thread
    processing_thread = Thread(target=process_frame)

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
        k = cv2.waitKey(5)
        if k == ord("q"):
            exit(1)

        # if the main thread is not alive
        # initial process and start
        if not processing_thread.is_alive():

            # while we get the return values
            while not return_values.empty():
                save_file(return_values)

            # set the main thread
            processing_thread = Thread(target=process_frame)

            # run the main processing thread
            processing_thread.start()

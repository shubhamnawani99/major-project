from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from itertools import repeat
from queue import Queue
from threading import Thread

import cv2
import matplotlib.pyplot as plt

from buffers import Buffer
from chopper import chop
from classifier import classify
from executor import process_and_upload

plt.rcParams["figure.figsize"] = [10, 6]

plt.ion()

fig, ax = plt.subplots()


def show_images(imgs):
    for i, img in enumerate(imgs):
        cv2.imshow('img{}'.format(i), img)


buffer = Buffer()
frame = None
cap = cv2.VideoCapture("../FinalCut.mp4")

retvals = Queue()


def process_frame():

    # chop the frame into different parts
    # to get the users from each frame
    imgs = chop(frame)

    # reset the buffer (the set of current people frame)
    buffer.reset_people()

    # upload the images and process them
    with ThreadPoolExecutor() as master:
        master.map(process_and_upload, imgs, repeat(buffer))

    # perform the classification
    classes, attentions = classify(buffer)

    # put the classification result in a queue
    retvals.put((classes, attentions))

    # set the presence of a person in the frame
    # if person not in the current frame,
    # assign false in the presence dict
    buffer.set_presences()


processingThread = Thread(target=process_frame)
NAMES = ["Ritesh Sethi", "Ayush Apoorva", "Nitin GL", "Vivek Chopra", "Shallen@GL", "Shreyan Datta Chakrabort"]
attentions = defaultdict(lambda: [])
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    if not processingThread.is_alive():
        ax.cla()
        while not retvals.empty():
            classes, scores = retvals.get()
            for key in scores:
                attentions[key].append(scores[key])
        processingThread = Thread(target=process_frame)
        processingThread.start()

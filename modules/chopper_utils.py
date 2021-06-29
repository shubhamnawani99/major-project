# import the libraries

import cv2
import numpy as np

# define the frame size
FRAME_SIZE = (1280, 720)

# store the different user screen layout that zoom provides
nums = [cv2.resize(cv2.imread('../resources/permutations/{}.png'.format(x), flags=cv2.IMREAD_GRAYSCALE), FRAME_SIZE)
        for x in [6, 7, 8]]


class Slicer(object):
    def __getitem__(self, idx):
        return idx


# get the height and width of the frame
h = FRAME_SIZE[1]
w = FRAME_SIZE[0]

''' 
cut the frame into different slices depending on the permutation used

     <------ w ------>
   ^   ---- ---- ----
   |  |    |    |    |  h/2
   h   ---- ---- ---- 
   |  |    |    |    |  h/2
   v   ---- ---- ----
        w/3  w/3  w/3
     
Permutation for a 6 user window
'''

slices = {
    6: [
        Slicer()[int((h / 2) - (h / 3)):int(h / 2), :int(w / 3)],
        Slicer()[int((h / 2) - (h / 3)):int(h / 2), int(w / 3):2 * int(w / 3)],
        Slicer()[int((h / 2) - (h / 3)):int(h / 2), 2 * int(w / 3):],
        Slicer()[int(h / 2):int((h / 2) + (h / 3)), :int(w / 3)],
        Slicer()[int(h / 2):int((h / 2) + (h / 3)), int(w / 3):2 * int(w / 3)],
        Slicer()[int(h / 2):int((h / 2) + (h / 3)):, 2 * int(w / 3):],
    ],
    7: [
        Slicer()[:int(h / 3), :int(w / 3)],
        Slicer()[:int(h / 3), int(w / 3):2 * int(w / 3)],
        Slicer()[:int(h / 3), 2 * int(w / 3):],
        Slicer()[int(h / 3):2 * int(h / 3), :int(w / 3)],
        Slicer()[int(h / 3):2 * int(h / 3), int(w / 3):2 * int(w / 3)],
        Slicer()[int(h / 3):2 * int(h / 3), 2 * int(w / 3):],
        Slicer()[int(2 * h / 3):, int(w / 3):2 * int(w / 3)],
    ],
    8: [
        Slicer()[:int(h / 3), :int(w / 3)],
        Slicer()[:int(h / 3), int(w / 3):2 * int(w / 3)],
        Slicer()[:int(h / 3), 2 * int(w / 3):],
        Slicer()[int(h / 3):2 * int(h / 3), :int(w / 3)],
        Slicer()[int(h / 3):2 * int(h / 3), int(w / 3):2 * int(w / 3)],
        Slicer()[int(h / 3):2 * int(h / 3), 2 * int(w / 3):],
        Slicer()[int(2 * h / 3):, int(w / 2) - (int(w / 3)):int(w / 2)],
        Slicer()[int(2 * h / 3):, int(w / 2):int(w / 2) + (int(w / 3))],
    ],
}


def chop_into(frame, n):
    cur_slices = slices[n]
    rets = []
    for slc in cur_slices:
        rets.append(frame[slc].copy())
    return rets


# main chop function
def chop(full_frame):

    # remove the black border from the full frame
    frame_color = full_frame[15:-15, 15:-15]

    # resize the frame into the FRAME_SIZE
    frame_color = cv2.resize(frame_color, FRAME_SIZE)

    # Convert to grayscale
    frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

    # determine the edges
    canny = cv2.Canny(frame, 50, 150, apertureSize=3)

    # apply the canny with the permutation filters
    confidence = [np.sum(cv2.bitwise_and(img, canny)) for img in nums]
    max_index = np.argmax(confidence) + 6

    # chop the whole frame into smaller frames
    # each frame consisting of 1 participant
    chopped_frames = chop_into(frame_color, max_index)

    return chopped_frames

import string
import cv2
import numpy as np
import pytesseract

# the path to installed pytesseract module goes here
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
translation = str.maketrans('', '', string.punctuation)


# prepare the frame
def prepare(frame):
    thresh = cv2.inRange(frame, (230, 230, 230), (255, 255, 255))
    return thresh


# perform threshold
def threshold(frame):
    # set element to black (0) if it lies below 180 or more than 255
    thresh = cv2.inRange(frame, (180, 180, 180), (255, 255, 255))
    # inverse color
    thresh = 255 - thresh
    return thresh


def crop2(binary):
    D = 10
    max_col_intensity = np.max(binary, 0)
    b = 0
    rightmost = True
    for i in range(len(max_col_intensity)):
        # 0 -> pure black
        if max_col_intensity[i] != 0:
            b = 0
        else:
            b += 1
        if b > D:
            rightmost = False
            break

    r = i if rightmost else i - D + int(D / 2)

    binary = binary[:, 0:r]
    max_row_intensity = np.max(binary, 1)
    start = -1
    end = -1
    for j, v in enumerate(max_row_intensity):
        if v == 255 and start == -1:
            start = j
        if v == 0 and start != -1:
            end = j
            break

    return start, end, r


def extract_name_from_frame(frame):
    # get the dimensions of frame
    height, width = frame.shape[0], frame.shape[1]

    # reduce the frame size
    # initial width crop to compensate for padding
    frame = frame[int(0.85 * height):, :int(0.5 * width)]

    # prepare the frame
    binary = prepare(frame)

    # crop down to select the name area
    t, b, r = crop2(binary)
    frame = frame[t - 5:b + 5, 0:r]
    t2 = threshold(frame)

    # convert the image to string using pytesseract
    s = pytesseract.image_to_string(t2).strip()

    return s.translate(translation)

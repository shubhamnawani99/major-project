from scipy.spatial import distance as dist

EYE_THRESHOLD = 0.24


def compute_eye_aspect_ratio(eye):
    #         [1]*   [2]*
    #    [0]*     eye     [3]*
    #         [5]*   [4]*

    # compute the euclidean distances from scipy library
    # between the two sets of vertical eye landmarks (x, y) coordinates
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    c = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    eye_ratio = (a + b) / (2.0 * c)

    # return the eye aspect ratio
    return eye_ratio


# returns the average eye aspect ratio of both eyes
def is_eye_open(kps):
    right_eye = kps[36:42]
    left_eye = kps[42:48]
    eye_ratio = (compute_eye_aspect_ratio(left_eye) + compute_eye_aspect_ratio(right_eye)) / 2.0
    # print(eye_ratio)
    # the eye is open if eye ratio more than the threshold
    return eye_ratio >= EYE_THRESHOLD


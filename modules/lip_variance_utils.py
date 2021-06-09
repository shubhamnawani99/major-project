# mouth -> (48, 68)
# inner_mouth -> (60, 68)
# right_eyebrow -> (17, 22)
# left_eyebrow -> (22, 27)
# right_eye -> (36, 42)
# left_eye -> (42, 48)
# nose -> (27, 36)
# jaw -> (0, 17)
from collections import deque

from scipy.spatial import distance as dist
import numpy as np


def get_total_dist(mouth):
    mean_pos = np.sum(mouth, axis=0) / 12
    # get (x, y) co-ordinates of the lip landmarks
    mean_pos = mean_pos[0:2]
    upper_lip = mouth[3][0:2]
    lower_lip = mouth[9][0:2]
    upper_lip_dist = dist.euclidean(mean_pos, upper_lip)
    lower_lip_dist = dist.euclidean(mean_pos, lower_lip)
    hor = dist.euclidean(mouth[0][0:2], mouth[6][0:2])
    return ((upper_lip_dist + lower_lip_dist) / hor) * 100


# get the lip variance
def get_lip_var(all_frames):
    distances = deque(maxlen=48)
    (start, end) = 48, 60  # (48,60) -> outer mouth, (60,68) -> inner mouth
    for shape in all_frames:
        mouth = shape[start:end]
        total_distance = get_total_dist(mouth)
        distances.append(total_distance)
    variance = np.var(distances)
    print("variance:", variance)
    return variance


# get the lip distance
def get_lip_dist(keypoints):
    mouth = keypoints[48:60]
    return get_total_dist(mouth)


# compute the lip variance
def get_lip_variance(distances):
    if len(distances) == 0:
        return 0
    variance = np.var(list(distances))
    return variance

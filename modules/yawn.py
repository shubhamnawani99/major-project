from scipy.spatial import distance as dist


def isYawn(kps):
    mouth = kps[48:60]
    left = mouth[0]
    right = mouth[6]
    top = mouth[3]
    bottom = mouth[9]
    hor = dist.euclidean(left, right)
    ver = dist.euclidean(top, bottom)
    ratio = ver / hor
    # print(ratio)
    if ratio >= 0.9:
        return True
    else:
        return False

from scipy.spatial import distance as dist


def is_yawning(kps):
    mouth = kps[48:60]

    left = mouth[0]
    right = mouth[6]
    top = mouth[3]
    bottom = mouth[9]

    euclidian_ratio_3d = is_yawing_euclidian(top, left, bottom, right)

    # print('Ratio: ', euclidian_ratio_3d)

    return euclidian_ratio_3d >= 0.9


def is_yawing_euclidian(top, left, bottom, right) -> float:
    horizontal_length = dist.euclidean(left, right)
    vertical_length = dist.euclidean(top, bottom)
    ratio = vertical_length / horizontal_length
    return ratio

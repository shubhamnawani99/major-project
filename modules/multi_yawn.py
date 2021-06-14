from scipy.spatial import distance as dist


def is_yawning_2d(kps_2d):
    mouth_2d = kps_2d[48:60]

    left_2d = mouth_2d[0]
    right_2d = mouth_2d[6]
    top_2d = mouth_2d[3]
    bottom_2d = mouth_2d[9]

    euclidian_ratio_2d = is_yawing_euclidian(top_2d, left_2d, bottom_2d, right_2d)

    print('2d: ', euclidian_ratio_2d)

    if euclidian_ratio_2d >= 0.9:
        return True
    else:
        return False


def is_yawning_3d(kps_3d):
    mouth_3d = kps_3d[48:60]
    mouth_2d_from_3d = [mouth_3d[idx][:2] for idx in range(0, 12)]

    left_3d = mouth_3d[0]
    right_3d = mouth_3d[6]
    top_3d = mouth_3d[3]
    bottom_3d = mouth_3d[9]

    left_2d_from_3d = mouth_2d_from_3d[0]
    right_2d_from_3d = mouth_2d_from_3d[6]
    top_2d_from_3d = mouth_2d_from_3d[3]
    bottom_2d_from_3dd = mouth_2d_from_3d[9]

    euclidian_ratio_3d = is_yawing_euclidian(top_3d, left_3d, bottom_3d, right_3d)
    euclidian_ratio_2d_from_3d = is_yawing_euclidian(top_2d_from_3d, left_2d_from_3d, bottom_2d_from_3dd,
                                                     right_2d_from_3d)
    minkowski_3 = is_yawing_minkowski(top_3d, left_3d, bottom_3d, right_3d, 3)
    print('3d: ', euclidian_ratio_3d, '2d from 3d: ', euclidian_ratio_2d_from_3d)

    if euclidian_ratio_3d >= 0.9:
        return True
    else:
        return False


def is_yawing_euclidian(top, left, bottom, right) -> float:
    horizontal_length = dist.euclidean(left, right)
    vertical_length = dist.euclidean(top, bottom)
    ratio = vertical_length / horizontal_length
    return ratio


def is_yawing_minkowski(top, left, bottom, right, modifier) -> float:
    horizontal_length = dist.minkowski(left, right, modifier)
    vertical_length = dist.minkowski(top, bottom, modifier)
    ratio = vertical_length / horizontal_length
    return ratio

def nearest_int(value, step=1):
    """
    Round to the nearest step
    :param value: value to round
    :param step: unit to round to
    """

    return int(round(value / step) * step)


def inside_bbox(x, y, bbox) -> bool:
    """
    Check if point is inside bounding box
    :param x: x coordinate of point
    :param y: y coordinate of point
    :param bbox: Tuple, list or array (x, y, w, h) of bounding box
    :return: bool
    """

    bx, by, bw, bh = bbox
    return bx <= x <= bx + bw and by <= y <= by + bh

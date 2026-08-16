def get_coordinates(dt_box):
    """
    从返回的检测框中获取坐标
    :param dt_box 检测框返回结果
    :return list 坐标点列表
    """
    coordinate_list = list()
    if isinstance(dt_box, list):
        for polygon in dt_box:
            points = list(polygon)
            if len(points) < 4:
                continue
            x_values = [int(point[0]) for point in points]
            y_values = [int(point[1]) for point in points]
            xmin = min(x_values)
            xmax = max(x_values)
            ymin = min(y_values)
            ymax = max(y_values)
            coordinate_list.append((xmin, xmax, ymin, ymax))
    return coordinate_list

import multiprocessing
import cv2
import numpy as np

from backend.config import config

def batch_generator(data, max_batch_size):
    """
    根据data大小，生成最大长度不超过max_batch_size的均匀批次数据
    """
    n_samples = len(data)
    # 尝试找到一个比MAX_BATCH_SIZE小的batch_size，以使得所有的批次数量尽量接近
    batch_size = max_batch_size
    num_batches = n_samples // batch_size

    # 处理最后一批可能不足batch_size的情况
    # 如果最后一批少于其他批次，则减小batch_size尝试平衡每批的数量
    while n_samples % batch_size < batch_size / 2.0 and batch_size > 1:
        batch_size -= 1  # 减小批次大小
        num_batches = n_samples // batch_size

    # 生成前num_batches个批次
    for i in range(num_batches):
        yield data[i * batch_size:(i + 1) * batch_size]

    # 将剩余的数据作为最后一个批次
    last_batch_start = num_batches * batch_size
    if last_batch_start < n_samples:
        yield data[last_batch_start:]

def create_mask(size, coords_list):
    mask = np.zeros(size, dtype="uint8")
    if coords_list:
        for coords in coords_list:
            xmin, xmax, ymin, ymax = coords
            # 为了避免框过小，放大10个像素
            x1 = xmin - config.subtitleAreaDeviationPixel.value
            if x1 < 0:
                x1 = 0
            y1 = ymin - config.subtitleAreaDeviationPixel.value
            if y1 < 0:
                y1 = 0
            x2 = xmax + config.subtitleAreaDeviationPixel.value
            y2 = ymax + config.subtitleAreaDeviationPixel.value
            cv2.rectangle(mask, (x1, y1),
                          (x2, y2), (255, 255, 255), thickness=-1)
    return mask


def create_text_box_mask(size, coords_list, padding=3):
    """Create a compact rectangular mask for trusted caption detections."""
    height, width = tuple(size)
    mask = np.zeros((height, width), dtype=np.uint8)
    padding = max(0, int(padding))
    for xmin, xmax, ymin, ymax in coords_list or []:
        x1 = max(0, min(width, int(xmin) - padding))
        x2 = max(0, min(width, int(xmax) + padding))
        y1 = max(0, min(height, int(ymin) - padding))
        y2 = max(0, min(height, int(ymax) + padding))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def _caption_area_for_box(box, caption_areas, frame_shape):
    height, width = frame_shape[:2]
    xmin, xmax, ymin, ymax = box
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    for area_ymin, area_ymax, area_xmin, area_xmax in caption_areas or []:
        if (
            area_ymin <= center_y <= area_ymax
            and area_xmin <= center_x <= area_xmax
        ):
            return (
                max(0, int(area_xmin)), min(width, int(area_xmax)),
                max(0, int(area_ymin)), min(height, int(area_ymax)),
            )
    return (0, width, 0, height)


def _expand_caption_line_box(frame, box, caption_areas):
    """Recover the full horizontal caption line around a partial OCR box."""
    height, width = frame.shape[:2]
    xmin, xmax, ymin, ymax = (int(value) for value in box)
    xmin, xmax = max(0, xmin), min(width, xmax)
    ymin, ymax = max(0, ymin), min(height, ymax)
    if xmax <= xmin or ymax <= ymin:
        return None

    text_height = max(4, ymax - ymin)
    area_xmin, area_xmax, area_ymin, area_ymax = _caption_area_for_box(
        box, caption_areas, frame.shape
    )
    vertical_padding = max(3, int(round(text_height * 0.35)))
    scan_ymin = max(area_ymin, ymin - vertical_padding)
    scan_ymax = min(area_ymax, ymax + vertical_padding)
    if scan_ymax <= scan_ymin or area_xmax <= area_xmin:
        return None

    roi = frame[scan_ymin:scan_ymax, area_xmin:area_xmax]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]
    blur_size = max(7, min(31, text_height | 1))
    background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    light_floor = max(140, int(np.percentile(gray, 68)))
    text_pixels = (
        (gray >= light_floor)
        & ((gray.astype(np.int16) - background.astype(np.int16)) >= 8)
        & (saturation <= 120)
    )
    minimum_column_pixels = max(2, int(round(roi.shape[0] * 0.10)))
    active_columns = (
        np.count_nonzero(text_pixels, axis=0) >= minimum_column_pixels
    ).astype(np.uint8)

    # Bridge spaces between letters/words, but do not blindly widen to the
    # complete selected rectangle. This is what recovers the two clipped ends
    # when OCR recognizes only the middle of a caption line.
    bridge = max(5, min(31, int(round(text_height * 0.9))))
    active_columns = cv2.morphologyEx(
        active_columns[None, :],
        cv2.MORPH_CLOSE,
        np.ones((1, bridge), dtype=np.uint8),
    )[0]
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        active_columns[None, :], connectivity=8
    )
    seed_xmin = max(0, xmin - area_xmin)
    seed_xmax = min(roi.shape[1], xmax - area_xmin)
    horizontal_slack = max(4, text_height)
    best = None
    for label in range(1, labels_count):
        run_xmin = int(stats[label, cv2.CC_STAT_LEFT])
        run_xmax = run_xmin + int(stats[label, cv2.CC_STAT_WIDTH])
        if run_xmax < seed_xmin - horizontal_slack:
            continue
        if run_xmin > seed_xmax + horizontal_slack:
            continue
        overlap = max(0, min(run_xmax, seed_xmax) - max(run_xmin, seed_xmin))
        distance = max(0, seed_xmin - run_xmax, run_xmin - seed_xmax)
        score = (overlap, -distance, run_xmax - run_xmin)
        if best is None or score > best[0]:
            best = (score, run_xmin, run_xmax)

    # OCR frequently returns only the middle word/fragment even though the
    # detector polygon visually overlaps that fragment well. Always guarantee
    # a wide horizontal safety margin; do not use it only as a fallback when
    # visual projection fails. 3.5 character-heights covers roughly 3--4 Latin
    # glyphs on either end, matching the common clipped-caption failure.
    fallback = max(16, int(round(text_height * 3.5)))
    if best is None:
        line_xmin = max(area_xmin, xmin - fallback)
        line_xmax = min(area_xmax, xmax + fallback)
    else:
        _, run_xmin, run_xmax = best
        line_xmin = max(area_xmin, area_xmin + run_xmin - 3)
        line_xmax = min(area_xmax, area_xmin + run_xmax + 3)
        # Never let a plausible-looking middle run suppress the deterministic
        # margin. That exact behavior left fragments on both caption edges.
        line_xmin = min(line_xmin, max(area_xmin, xmin - fallback))
        line_xmax = max(line_xmax, min(area_xmax, xmax + fallback))

    return (
        line_xmin, line_xmax,
        max(area_ymin, ymin - vertical_padding),
        min(area_ymax, ymax + vertical_padding),
    )


def create_caption_text_union_mask(
    frame_coordinate_pairs, caption_areas, size=None
):
    """Mask complete caption lines even when OCR returns only a middle box."""
    pairs = list(frame_coordinate_pairs or [])
    if size is None:
        if not pairs:
            raise ValueError("size is required when no caption references exist")
        size = pairs[0][0].shape[:2]
    expanded_boxes = []
    for frame, coordinates in pairs:
        for coordinate in coordinates or []:
            expanded = _expand_caption_line_box(
                frame, coordinate, caption_areas
            )
            if expanded is not None and expanded not in expanded_boxes:
                expanded_boxes.append(expanded)
    return create_text_box_mask(size, expanded_boxes, padding=2)


def _add_chinese_text_candidates(mask, frame, coords_list):
    """Add conservative stroke candidates from one reference frame."""
    height, width = frame.shape[:2]
    for xmin, xmax, ymin, ymax in coords_list or []:
        xmin = max(0, min(int(xmin), width))
        xmax = max(0, min(int(xmax), width))
        ymin = max(0, min(int(ymin), height))
        ymax = max(0, min(int(ymax), height))
        if xmax - xmin < 24 or ymax - ymin < 10:
            continue

        roi = frame[ymin:ymax, xmin:xmax]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        saturation = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]
        kernel = max(7, ((max(3, (ymax - ymin) // 3) // 2) * 2) + 1)
        local_background = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        gray_i16 = gray.astype(np.int16)
        local_i16 = local_background.astype(np.int16)

        bright_floor = max(155, int(np.percentile(gray, 72)))
        neutral_color = saturation <= 85
        bright_strokes = (
            (gray >= bright_floor)
            & ((gray_i16 - local_i16) >= 12)
            & neutral_color
        )
        candidate = bright_strokes.astype(np.uint8) * 255
        candidate = cv2.morphologyEx(
            candidate, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)
        )
        evidence_coverage = np.count_nonzero(candidate) / float(candidate.size)
        columns = np.flatnonzero(np.any(candidate > 0, axis=0))
        if evidence_coverage < 0.005 or evidence_coverage > 0.50 or len(columns) < 2:
            continue
        if columns[-1] - columns[0] < max(12, int(candidate.shape[1] * 0.25)):
            continue
        candidate = cv2.dilate(
            candidate, np.ones((3, 3), np.uint8), iterations=4
        )
        # OCR boxes are commonly tight around individual glyph strokes. Join
        # nearby strokes before temporal union so the reconstructed patch does
        # not retain recognizable holes or broken pieces of the original text.
        # This is still constrained to a confirmed Chinese OCR rectangle.
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5)),
        )
        mask[ymin:ymax, xmin:xmax] = cv2.bitwise_or(
            mask[ymin:ymax, xmin:xmax], candidate
        )


def create_chinese_text_union_mask(frame_coordinate_pairs, size=None):
    """Build one mask from multiple temporal references without full-frame churn."""
    pairs = list(frame_coordinate_pairs or [])
    if size is None:
        if not pairs:
            raise ValueError("size is required when no Chinese mask references exist")
        size = pairs[0][0].shape[:2]
    mask = np.zeros(tuple(size), dtype=np.uint8)
    for reference_frame, coords_list in pairs:
        _add_chinese_text_candidates(mask, reference_frame, coords_list)

    if np.any(mask):
        # The OCR rectangle is tight and the first dilation above is clipped to
        # it. This final three-pixel halo covers anti-aliased edges and shadows
        # just outside the rectangle without replacing the entire box.
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
    return mask


def create_chinese_text_mask(frame, coords_list):
    """Build a conservative stroke-level mask around confirmed Chinese boxes.

    Full rectangular OCR boxes can cover diagrams, logos, or small components
    behind the text. This mask selects high-contrast, low-saturation light
    strokes and deliberately skips ambiguous, dark, or colored regions.
    Missing an unusual text color is safer than deleting real video content.
    """
    return create_chinese_text_union_mask([(frame, coords_list)])


def create_feathered_mask_alpha(mask, radius=None):
    """Return a hard-core mask with a short, soft outer transition.

    Confirmed text pixels remain fully replaced (alpha 1). Only a small halo
    outside the binary mask is blended, avoiding a glyph-shaped hard seam while
    keeping unrelated frame detail bit-identical farther away.
    """
    if mask.ndim != 2:
        raise ValueError("feather mask must be a 2-D array")
    height, width = mask.shape
    if radius is None:
        radius = max(1, min(4, int(round(min(height, width) / 360.0))))
    radius = max(0, int(radius))
    hard = (mask > 0).astype(np.float32)
    if radius == 0 or not np.any(hard):
        return hard

    # Give the replacement one solid pixel outside the detected edge, then
    # fade it into the source over a few pixels. np.maximum preserves alpha=1
    # over the complete original mask.
    solid_halo = cv2.dilate(
        hard,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    kernel_size = radius * 2 + 1
    feather = cv2.GaussianBlur(
        solid_halo, (kernel_size, kernel_size), sigmaX=max(0.8, radius * 0.65)
    )
    return np.maximum(hard, np.clip(feather, 0.0, 1.0))


def composite_with_feather(original, generated, mask, radius=None):
    """Blend an inpainted image through a feathered Chinese-text mask."""
    if original.shape != generated.shape or original.shape[:2] != mask.shape:
        raise ValueError("original, generated, and mask dimensions must match")
    alpha = create_feathered_mask_alpha(mask, radius=radius)[:, :, None]
    if not np.any(alpha):
        return original.copy()
    blended = (
        generated.astype(np.float32) * alpha
        + original.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)

def get_inpaint_area_by_mask(W, H, h, mask, multiple=1):
    """
    获取字幕去除区域，根据mask来确定需要填补的区域和高度，
    并根据模型要求调整区域大小为指定倍数
    
    Args:
        W: 图像宽度
        H: 图像高度
        h: 检测区域高度
        mask: 遮罩图像
        multiple: 区域尺寸需要满足的倍数，默认为1
    
    Returns:
        调整后的绘画区域列表，格式为[(ymin, ymax, xmin, xmax), ...]
    """
    # 存储绘画区域的列表
    inpaint_area = []
    
    # 如果mask全为0，直接返回空列表
    if np.all(mask == 0):
        return inpaint_area
    
    # 使用连通组件分析找出mask中的所有孤岛
    # 首先确保mask是二值图像
    binary_mask = (mask > 0).astype(np.uint8) * 255
    
    # 查找连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # 跳过背景（标签0）
    island_info = []
    for i in range(1, num_labels):
        # 获取当前孤岛的统计信息
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # 忽略太小的区域（可能是噪点）
        if area < 10:
            continue
        
        # 保存孤岛信息：顶部y坐标，底部y坐标，中心点y坐标，面积，标签
        center_y = int(centroids[i][1])
        island_info.append((y, y + height, center_y, area, i))
    
    # 如果没有有效孤岛，返回空列表
    if not island_info:
        return inpaint_area
    
    # 按中心点y坐标排序孤岛
    island_info.sort(key=lambda x: x[2])
    
    # 尝试合并孤岛
    merged_islands = []
    current_group = [island_info[0]]
    
    for i in range(1, len(island_info)):
        # 当前组的范围
        min_y = min([island[0] for island in current_group])
        max_y = max([island[1] for island in current_group])
        
        # 当前孤岛
        top_y, bottom_y, center_y, _, _ = island_info[i]
        
        # 计算如果添加当前孤岛，新组的范围
        new_min_y = min(min_y, top_y)
        new_max_y = max(max_y, bottom_y)
        
        # 检查是否有mask连接当前组和新孤岛
        has_connection = False
        if max_y < top_y:  # 只有当前组在新孤岛上方时才需要检查连接
            # 检查两个区域之间是否有mask像素
            middle_region = binary_mask[max_y:top_y, :]
            if np.any(middle_region > 0):
                has_connection = True
        else:  # 重叠或相邻
            has_connection = True
        
        # 检查合并后的高度是否在h范围内，并且有连接
        if new_max_y - new_min_y <= h and has_connection:
            # 可以合并
            current_group.append(island_info[i])
        else:
            # 无法合并，保存当前组并开始新组
            merged_islands.append(current_group)
            current_group = [island_info[i]]
    
    # 添加最后一个组
    merged_islands.append(current_group)
    
    # 为每个合并后的组创建区域
    for group in merged_islands:
        # 获取组内所有孤岛的范围
        min_y = min([island[0] for island in group])
        max_y = max([island[1] for island in group])
        
        # 计算组的中心点
        center_y = sum([island[2] for island in group]) // len(group)
        
        # 确保区域高度精确等于h
        half_h = h // 2
        
        # 从中心点向上下扩展，确保高度为h
        ymin = max(0, center_y - half_h)
        ymax = ymin + h  # 确保高度精确等于h
        
        # 如果超出图像底部，从底部向上调整
        if ymax > H:
            ymax = H
            ymin = max(0, H - h)  # 确保高度为h
        
        # 检查是否包含了所有孤岛
        if ymin > min_y or ymax < max_y:
            # 如果区域不能完全包含所有孤岛，尝试调整位置但保持高度为h
            if max_y - min_y <= h:
                # 孤岛总高度不超过h，可以调整位置使其完全包含
                ymin = min_y
                ymax = ymin + h
                # 如果超出底部，从底部向上调整
                if ymax > H:
                    ymax = H
                    ymin = max(0, H - h)
            else:
                # 孤岛总高度超过h，无法完全包含，优先包含中心区域
                # 计算孤岛的中心
                island_center = (min_y + max_y) // 2
                ymin = max(0, island_center - half_h)
                ymax = ymin + h
                # 如果超出底部，从底部向上调整
                if ymax > H:
                    ymax = H
                    ymin = max(0, H - h)
        
        # 使用完整宽度
        xmin = 0
        xmax = W
        
        # 调整区域大小为指定倍数
        if multiple > 1:
            # 计算区域高度
            height = ymax - ymin
            # 计算需要调整的高度，使其成为multiple的倍数
            remainder = height % multiple
            
            if remainder != 0:
                # 需要调整的像素数
                adjust_pixels = multiple - remainder
                
                # 计算区域中心点
                center_y = (ymin + ymax) / 2
                
                # 优先对称扩展
                if ymin - adjust_pixels/2 >= 0 and ymax + adjust_pixels/2 <= H:
                    # 对称扩展
                    ymin = int(center_y - height/2 - adjust_pixels/2)
                    ymax = int(center_y + height/2 + adjust_pixels/2)
                # 如果对称扩展会超出边界，尝试对称缩小
                elif height > multiple:  # 确保缩小后高度至少为multiple
                    # 对称缩小
                    ymin = int(center_y - (height - remainder)/2)
                    ymax = int(center_y + (height - remainder)/2)
                # 如果无法对称调整，则尝试单边调整
                else:
                    # 向下扩展
                    if ymax + adjust_pixels <= H:
                        ymax += adjust_pixels
                    # 向上扩展
                    elif ymin - adjust_pixels >= 0:
                        ymin -= adjust_pixels
                    # 如果都不行，则尝试缩小区域
                    elif height > multiple:
                        ymax = ymin + height - remainder
            
            # 调整宽度，确保是multiple的倍数
            width = xmax - xmin
            remainder_w = width % multiple
            
            if remainder_w != 0:
                # 需要调整的像素数
                adjust_pixels_w = multiple - remainder_w
                
                # 计算中心点，对称缩小
                center_x = (xmin + xmax) / 2
                xmin = int(center_x - (width - remainder_w)/2)
                xmax = int(center_x + (width - remainder_w)/2)
        
        # 将该区域添加到列表中，格式为(ymin, ymax, xmin, xmax)
        area = (int(ymin), int(ymax), int(xmin), int(xmax))
        if area not in inpaint_area:
            inpaint_area.append(area)
    
    return inpaint_area  # 返回绘画区域列表，格式为[(ymin, ymax, xmin, xmax), ...]
    
def expand_frame_ranges(frame_ranges, backward_frame_count, forward_frame_count):
    """
    扩展帧区间列表，向前和向后扩展指定的帧数，并确保区间连续性
    
    Args:
        frame_ranges: 帧区间列表，格式为[(start1, end1), (start2, end2), ...]
        backward_frame_count: 向前扩展的帧数
        forward_frame_count: 向后扩展的帧数
        
    Returns:
        扩展后的帧区间列表，保证连续性
    """
    if not frame_ranges:
        return []
    
    # 按起始帧排序
    sorted_ranges = sorted(frame_ranges)
    expanded_ranges = []
    
    for i, (start, end) in enumerate(sorted_ranges):
        # 向前扩展，但不能小于1
        new_start = max(1, start - backward_frame_count)
        
        # 向后扩展
        new_end = end + forward_frame_count
        
        # 检查是否与下一个区间重叠
        if i < len(sorted_ranges) - 1:
            next_start = sorted_ranges[i + 1][0]
            
            # 如果扩展后的结束帧超过了下一个区间的起始帧
            if new_end >= next_start:
                # 计算中点
                mid_point = (end + next_start) // 2
                
                # 如果区间是连续的(相差1)，则对半平分
                if next_start - end == 1:
                    new_end = end  # 保持原结束帧
                else:
                    # 非连续区间，限制扩展到下一个区间起始帧减去backward_frame_count
                    max_expand = next_start - 1  # 确保不会与下一个区间重叠
                    new_end = min(new_end, max_expand)
        
        # 确保与前一个区间不重叠
        if expanded_ranges:
            prev_end = expanded_ranges[-1][1]
            if new_start <= prev_end:
                # 如果新区间的开始小于等于前一个区间的结束，调整开始位置
                new_start = prev_end + 1
        
        # 确保区间有效（开始不大于结束）
        if new_start <= new_end:
            expanded_ranges.append((new_start, new_end))
        else:
            # 如果调整后区间无效，保留原始区间
            expanded_ranges.append((start, end))
    
    return expanded_ranges

def is_frame_number_in_ab_sections(frame_no, ab_sections):
    """
    检查给定的帧号是否在指定的A/B区间内。

    Args:
        frame_no: 要检查的帧号
        ab_sections: 包含A/B区间的列表，格式为[range(start, end), ...]

    Returns:
        如果帧号在A/B区间内，返回True；否则返回False。
    """
    if ab_sections is None:
        return True
    if len(ab_sections) <= 0:
        return True
    for section in ab_sections:
        if frame_no in section:
            return True
    return False

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")

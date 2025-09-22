def iou(box1, box2):
    """
    计算两个矩形的交并比（IoU）。
    参数
    ----
    box1, box2: tuple or list
        长度为 4，格式 (x1, y1, x2, y2) —— 任意对角顶点即可。
    返回
    ----
    float: [0, 1] 区间的 IoU 值。无重叠时返回 0。
    """
    # 规范化：保证 x1<x2, y1<y2
    x1_min, x1_max = sorted(box1[::2])
    y1_min, y1_max = sorted(box1[1::2])
    x2_min, x2_max = sorted(box2[::2])
    y2_min, y2_max = sorted(box2[1::2])

    # 交集
    inter_x1 = max(x1_min, x2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y1 = max(y1_min, y2_min)
    inter_y2 = min(y1_max, y2_max)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # 并集
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    # 防止除零
    return inter_area / union_area if union_area > 0 else 0.0


# ------------------ 测试 ------------------
if __name__ == "__main__":
    pred = (10, 10, 50, 50)
    gt   = (30, 30, 70, 70)
    print("IoU =", iou(pred, gt))  # 0.14285714285714285
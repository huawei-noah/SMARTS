import math
import warnings

import numpy as np

warnings.filterwarnings("ignore")

eps = 1e-6
max_time_diff = 8
max_dist_diff = 30
vehicle_w = 0.3


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Line:
    def __init__(self, m, n):
        self.m = m
        self.n = n


def AddPoint(P1, P2):
    return Point(P1.x + P2.x, P1.y + P2.y)


def MinusPoint(P1, P2):
    return Point(P1.x - P2.x, P1.y - P2.y)


def NumMultiPoint(P1, num):
    return Point(P1.x * num, P1.y * num)


def Cross(P1, P2):
    return P1.x * P2.y - P1.y * P2.x


def Dot(P1, P2):
    return P1.x * P2.x + P1.y * P2.y


def IsPointInPoly(P, poly):
    px, py = P.x, P.y
    InPoly = False
    for i in range(4):
        sx, sy = poly[i].x, poly[i].y
        tx, ty = poly[i + 1].x, poly[i + 1].y
        # on vertex
        if (px == sx and py == ty) or (px == tx and py == ty):
            return True
        # relation with edge
        if (sy < py and ty >= py) or (sy >= py and ty < py):
            x = sx + (py - sy) * (tx - sx) / (ty - sy)
            if x == px:
                return True
            if x > px:
                InPoly = ~InPoly
    return InPoly


def IsFormalCross(P1, P2, P3, P4):
    return (
        Cross(MinusPoint(P2, P1), MinusPoint(P3, P1))
        * Cross(MinusPoint(P2, P1), MinusPoint(P4, P1))
        < -eps
        and Cross(MinusPoint(P4, P3), MinusPoint(P1, P3))
        * Cross(MinusPoint(P4, P3), MinusPoint(P2, P3))
        < -eps
    )


def IsVehicleCross(M1, N1, M2, N2):
    """
    Considering the area of vehicle, they got intersection as long as two rectangles overlap,
    and the intersections are their center points.
    """

    def draw_rect(M, N):
        theta = math.atan2(N.y - M.y, N.x - M.x)
        A = AddPoint(
            M, Point(vehicle_w * math.sin(theta), -vehicle_w * math.cos(theta))
        )
        B = AddPoint(A, MinusPoint(N, M))
        D = AddPoint(
            M, Point(-vehicle_w * math.sin(theta), vehicle_w * math.cos(theta))
        )
        C = AddPoint(D, MinusPoint(N, M))
        v = [A, B, C, D, A]
        return v

    v1 = draw_rect(M1, N1)
    v2 = draw_rect(M2, N2)
    # if two rectangles overlap, there are points in a rectangle or at the intersection of edges
    for i in range(4):
        for _ in range(4):
            if IsFormalCross(v1[i], v1[i + 1], v2[i], v2[i + 1]):
                return True
    for i in range(4):
        if IsPointInPoly(v1[i], v2):
            return True
        if IsPointInPoly(v2[i], v1):
            return True
    return False


def CountIntersect(L1, L2):
    area1 = Cross(MinusPoint(L1.m, L2.m), MinusPoint(L2.n, L2.m)) / 2
    area2 = Cross(MinusPoint(L2.n, L2.m), MinusPoint(L1.n, L2.m)) / 2
    msg = None
    Pt = AddPoint(L1.m, NumMultiPoint(MinusPoint(L1.n, L1.m), area1 / (area1 + area2)))
    if -eps < area1 + area2 < eps:
        Pt = NumMultiPoint(AddPoint(L1.n, L2.m), 0.5)
        msg = "parallel"
    if Dot(MinusPoint(L2.m, L2.n), MinusPoint(L2.m, Pt)) > 0:
        Pt = NumMultiPoint(AddPoint(L1.n, L2.m), 0.5)
        msg = "convex"
    return Pt, msg


def CountSegIntersect(A1, B1, A2, B2):
    area1 = Cross(MinusPoint(A1, A2), MinusPoint(B2, A2)) / 2
    area2 = Cross(MinusPoint(B2, A2), MinusPoint(B1, A2)) / 2
    P = AddPoint(A1, NumMultiPoint(MinusPoint(B1, A1), area1 / (area1 + area2)))
    return P


def CountPointDist(P1, P2):
    return math.sqrt(math.pow(P1.x - P2.x, 2) + math.pow(P1.y - P2.y, 2))


def Count_dtimespeed(v0, v1, s1, s2):
    return s1 / s2


class Circle:
    def __init__(self, t0, t1, t2):
        a = CountPointDist(t0, t1)
        b = CountPointDist(t1, t2)
        c = CountPointDist(t0, t2)
        tarea = math.fabs(Cross(MinusPoint(t0, t1), MinusPoint(t1, t2)) / 2)
        self.r = a * b * c / tarea / 4
        c1 = (t0.x * t0.x + t0.y * t0.y - t1.x * t1.x - t1.y * t1.y) / 2
        c2 = (t0.x * t0.x + t0.y * t0.y - t2.x * t2.x - t2.y * t2.y) / 2
        self.cx = (c1 * (t0.y - t2.y) - c2 * (t0.y - t1.y)) / (
            (t0.x - t1.x) * (t0.y - t2.y) - (t0.x - t2.x) * (t0.y - t1.y)
        )
        self.cy = (c1 * (t0.x - t2.x) - c2 * (t0.x - t1.x)) / (
            (t0.y - t1.y) * (t0.x - t2.x) - (t0.y - t2.y) * (t0.x - t1.x)
        )

    def GetPoint(self, theta):
        return self.cx + math.cos(theta) * self.r, self.cy + math.sin(theta) * self.r


# end geometry functions
def count_intersection_area(pos_arr1, pos_arr2):
    # find the intersection of two lists
    min_x1, max_x1 = (
        np.min(pos_arr1[:, 0]) - vehicle_w,
        np.max(pos_arr1[:, 0]) + vehicle_w,
    )
    min_y1, max_y1 = (
        np.min(pos_arr1[:, 1]) - vehicle_w,
        np.max(pos_arr1[:, 1]) + vehicle_w,
    )
    min_x2, max_x2 = (
        np.min(pos_arr2[:, 0]) - vehicle_w,
        np.max(pos_arr2[:, 0]) + vehicle_w,
    )
    min_y2, max_y2 = (
        np.min(pos_arr2[:, 1]) - vehicle_w,
        np.max(pos_arr2[:, 1]) + vehicle_w,
    )
    min_x, max_x = max(min_x1, min_x2), min(max_x1, max_x2)
    min_y, max_y = max(min_y1, min_y2), min(max_y1, max_y2)
    # reserve the points next to the intersection in case that they are end points of the target segment
    def cnt_idx(pos_arr_a, min_n, max_n):
        idx_n = (pos_arr_a >= min_n) * (pos_arr_a <= max_n)
        if ~idx_n.any():
            idx_n_min = pos_arr_a >= min_n
            idx_n_min = (
                idx_n_min
                + np.append(idx_n_min[1:], False)
                + np.insert(idx_n_min[:-1], 0, False)
            )
            idx_n_max = pos_arr_a <= max_n
            idx_n_max = (
                idx_n_max
                + np.append(idx_n_max[1:], False)
                + np.insert(idx_n_max[:-1], 0, False)
            )
            idx_n = idx_n_min * idx_n_max
        return idx_n

    proc_idx1 = cnt_idx(pos_arr1[:, 0], min_x, max_x) * cnt_idx(
        pos_arr1[:, 1], min_y, max_y
    )
    proc_idx2 = cnt_idx(pos_arr2[:, 0], min_x, max_x) * cnt_idx(
        pos_arr2[:, 1], min_y, max_y
    )
    proc_idx1 = (
        proc_idx1
        + np.append(proc_idx1[1:], False)
        + np.insert(proc_idx1[:-1], 0, False)
    )
    proc_idx2 = (
        proc_idx2
        + np.append(proc_idx2[1:], False)
        + np.insert(proc_idx2[:-1], 0, False)
    )
    # save reserved points
    def proc_list(pos_arr, proc_idx):
        pos_list = []
        part_tmp = []
        for (pos, res) in zip(pos_arr, proc_idx):
            if res:
                part_tmp.append(pos.tolist())
            elif len(part_tmp) > 0:
                pos_list.append(part_tmp)
                part_tmp = []
        if len(part_tmp) > 0:
            pos_list.append(part_tmp)
        return pos_list

    pos_list1 = proc_list(pos_arr1, proc_idx1)
    pos_list2 = proc_list(pos_arr2, proc_idx2)

    return pos_list1, pos_list2


def count_intersection_segment(pos_list1, pos_list2):
    # Traverse every pair to find the target segment
    for traj1 in pos_list1:
        for pos1 in range(len(traj1) - 1):
            A1, B1 = (
                Point(traj1[pos1][0], traj1[pos1][1]),
                Point(traj1[pos1 + 1][0], traj1[pos1 + 1][1]),
            )
            for traj2 in pos_list2:
                for pos2 in range(len(traj2) - 1):
                    A2, B2 = (
                        Point(traj2[pos2][0], traj2[pos2][1]),
                        Point(traj2[pos2 + 1][0], traj2[pos2 + 1][1]),
                    )
                    if IsFormalCross(A1, B1, A2, B2):
                        P = CountSegIntersect(A1, B1, A2, B2)
                        return A1, B1, A2, B2, P, P
                    if IsVehicleCross(A1, B1, A2, B2):
                        P1 = NumMultiPoint(AddPoint(A1, B1), 0.5)
                        P2 = NumMultiPoint(AddPoint(A2, B2), 0.5)
                        return A1, B1, A2, B2, P1, P2

    raise Exception("No target segment found.")


def generate_curve(A, B, C, D):
    """
    if Seg(A1, B1) and Seg(A2, B2) intersects at P,
    then C is the closest point to A in the trajectory1 opposite to the direction of B,
    and D is the closest point to B opposite to the direction of A

    output: Bezier curve
    """
    line_CA = Line(C, A)
    line_BD = Line(B, D)
    Pt, msg = CountIntersect(line_CA, line_BD)
    curve = []
    curve.append(Point(A.x, A.y))
    curve.append(Point(2 * (Pt.x - A.x), 2 * (Pt.y - A.y)))
    curve.append(Point(B.x - curve[0].x - curve[1].x, B.y - curve[0].y - curve[1].y))
    return curve, msg


def count_time_score(time_arr1, time_arr2, idx_A1, idx_A2, t1, t2):
    arrtime_1 = time_arr1[idx_A1] + (time_arr1[idx_A1 + 1] - time_arr1[idx_A1]) * t1
    arrtime_2 = time_arr2[idx_A2] + (time_arr2[idx_A2 + 1] - time_arr2[idx_A2]) * t2
    if arrtime_1 < arrtime_2:
        return 1, arrtime_1, arrtime_2 - arrtime_1
    return 2, arrtime_2, arrtime_2 - arrtime_1


def count_dist_score(pos_arr, time_arr, arrtime, P):
    curr_idx = np.where(time_arr < arrtime)[0][-1]
    assert curr_idx > 0 and curr_idx < pos_arr.shape[0] - 1
    dt = (arrtime - time_arr[curr_idx]) / (time_arr[curr_idx + 1] - time_arr[curr_idx])
    C_, A_, B_, D_ = (
        pos_arr[curr_idx - 1],
        pos_arr[curr_idx],
        pos_arr[curr_idx + 1],
        pos_arr[curr_idx + 2],
    )
    C, A, B, D = (
        Point(C_[0], C_[1]),
        Point(A_[0], A_[1]),
        Point(B_[0], B_[1]),
        Point(D_[0], D_[1]),
    )
    curve, _ = generate_curve(A, B, C, D)
    pos_x = curve[0].x + curve[1].x * dt + curve[2].x * dt * dt
    pos_y = curve[0].y + curve[1].y * dt + curve[2].y * dt * dt
    Q = Point(pos_x, pos_y)
    dist_diff = CountPointDist(P, Q)
    return Q, dist_diff


def predict_traj(pos_arr1, pos_arr2, speed_arr1, speed_arr2, time_arr1, time_arr2):
    min_x1, max_x1 = np.min(pos_arr1[:, 0]), np.max(pos_arr1[:, 0])
    min_y1, max_y1 = np.min(pos_arr1[:, 1]), np.max(pos_arr1[:, 1])
    min_x2, max_x2 = np.min(pos_arr2[:, 0]), np.max(pos_arr2[:, 0])
    min_y2, max_y2 = np.min(pos_arr2[:, 1]), np.max(pos_arr2[:, 1])
    min_x, max_x = max(min_x1, min_x2), min(max_x1, max_x2)
    min_y, max_y = max(min_y1, min_y2), min(max_y1, max_y2)
    max_xx, min_xx = max(max_x1, max_x2), min(min_x1, min_x2)
    max_yy, min_yy = max(max_y1, max_y2), min(min_y1, min_y2)
    # predict the traj
    def predict_straight(pos_arr, speed_arr, time_arr, dimsign):
        pos0 = len(pos_arr[:, 0]) - 1
        i = 0
        if dimsign == "xr":
            nx = pos_arr[-1, 0]
            while nx < max_xx:
                j = i % 12
                nx = (
                    pos_arr[pos0 + i, 0]
                    + pos_arr[pos0 - 11 + j, 0]
                    - pos_arr[pos0 - 12 + j, 0]
                )
                ny = (
                    pos_arr[pos0 + i, 1]
                    + pos_arr[pos0 - 11 + j, 1]
                    - pos_arr[pos0 - 12 + j, 1]
                )
                pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
                time_arr = np.append(
                    time_arr,
                    time_arr[pos0 + i]
                    + time_arr[pos0 - 11 + j]
                    - time_arr[pos0 - 12 + j],
                )
                speed_arr = np.append(speed_arr, speed_arr[pos0])
                i += 1
        elif dimsign == "xl":
            nx = pos_arr[-1, 0]
            while nx > min_xx:
                j = i % 12
                nx = (
                    pos_arr[pos0 + i, 0]
                    + pos_arr[pos0 - 11 + j, 0]
                    - pos_arr[pos0 - 12 + j, 0]
                )
                ny = (
                    pos_arr[pos0 + i, 1]
                    + pos_arr[pos0 - 11 + j, 1]
                    - pos_arr[pos0 - 12 + j, 1]
                )
                pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
                time_arr = np.append(
                    time_arr,
                    time_arr[pos0 + i]
                    + time_arr[pos0 - 11 + j]
                    - time_arr[pos0 - 12 + j],
                )
                speed_arr = np.append(speed_arr, speed_arr[pos0])
                i += 1
        elif dimsign == "yu":
            ny = pos_arr[-1, 1]
            while ny < max_yy:
                j = i % 12
                nx = (
                    pos_arr[pos0 + i, 0]
                    + pos_arr[pos0 - 11 + j, 0]
                    - pos_arr[pos0 - 12 + j, 0]
                )
                ny = (
                    pos_arr[pos0 + i, 1]
                    + pos_arr[pos0 - 11 + j, 1]
                    - pos_arr[pos0 - 12 + j, 1]
                )
                pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
                time_arr = np.append(
                    time_arr,
                    time_arr[pos0 + i]
                    + time_arr[pos0 - 11 + j]
                    - time_arr[pos0 - 12 + j],
                )
                speed_arr = np.append(speed_arr, speed_arr[pos0])
                i += 1
        else:
            ny = pos_arr[-1, 1]
            while ny > min_yy:
                j = i % 12
                nx = (
                    pos_arr[pos0 + i, 0]
                    + pos_arr[pos0 - 11 + j, 0]
                    - pos_arr[pos0 - 12 + j, 0]
                )
                ny = (
                    pos_arr[pos0 + i, 1]
                    + pos_arr[pos0 - 11 + j, 1]
                    - pos_arr[pos0 - 12 + j, 1]
                )
                pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
                time_arr = np.append(
                    time_arr,
                    time_arr[pos0 + i]
                    + time_arr[pos0 - 11 + j]
                    - time_arr[pos0 - 12 + j],
                )
                speed_arr = np.append(speed_arr, speed_arr[pos0])
                i += 1
        j = i % 12
        nx = (
            pos_arr[pos0 + i, 0] + pos_arr[pos0 - 11 + j, 0] - pos_arr[pos0 - 12 + j, 0]
        )
        ny = (
            pos_arr[pos0 + i, 1] + pos_arr[pos0 - 11 + j, 1] - pos_arr[pos0 - 12 + j, 1]
        )
        pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
        time_arr = np.append(
            time_arr,
            time_arr[pos0 + i] + time_arr[pos0 - 11 + j] - time_arr[pos0 - 12 + j],
        )
        speed_arr = np.append(speed_arr, speed_arr[pos0])
        return pos_arr, speed_arr, time_arr

    def predict_turn(pos_arr, speed_arr, time_arr, dimsign, keypoint):
        # circumcircle
        pos0 = len(pos_arr[:, 0]) - 1
        pos1 = (pos0 + keypoint) // 2
        c = Circle(
            Point(pos_arr[pos0][0], pos_arr[pos0][1]),
            Point(pos_arr[pos1][0], pos_arr[pos1][1]),
            Point(pos_arr[keypoint][0], pos_arr[keypoint][1]),
        )
        theta0 = math.atan2(pos_arr[pos0][1], pos_arr[pos0][0])
        dtheta = theta0 - math.atan2(pos_arr[pos0 - 1][1], pos_arr[pos0 - 1][0])
        i = 0
        lx, ly = pos_arr[-1, 0], pos_arr[-1, 1]
        dx, dy = 2, 2
        while dx > 0.005 and dy > 0.005:
            nx, ny = c.GetPoint(theta0 + dtheta * i)
            dx, dy = math.fabs(nx - lx), math.fabs(ny - ly)
            lx, ly = nx, ny
            pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
            time_arr = np.append(
                time_arr, time_arr[pos0 + i] + time_arr[pos0] - time_arr[pos0 - 1]
            )
            speed_arr = np.append(speed_arr, speed_arr[pos0])
            i += 1
        nx, ny = c.GetPoint(theta0 + dtheta * i)
        pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
        time_arr = np.append(
            time_arr, time_arr[pos0 + i] + time_arr[pos0] - time_arr[pos0 - 1]
        )
        speed_arr = np.append(speed_arr, speed_arr[pos0])
        if dimsign == "xr":
            if theta0 + dtheta * i > -0.2 and theta0 + dtheta * i < 0.2:
                if dtheta > 0:
                    dimsign = "yu"
                else:
                    dimsign = "yd"
        if dimsign == "xl":
            if theta0 + dtheta * i < -2.9 or theta0 + dtheta * i > 2.9:
                if dtheta > 0:
                    dimsign = "yd"
                else:
                    dimsign = "yu"
        if dimsign == "yu":
            if theta0 + dtheta * i > 1.4 and theta0 + dtheta * i < 1.8:
                if dtheta > 0:
                    dimsign = "xl"
                else:
                    dimsign = "xr"
        if dimsign == "yd":
            if theta0 + dtheta * i < -1.4 and theta0 + dtheta * i > -1.8:
                if dtheta > 0:
                    dimsign = "xr"
                else:
                    dimsign = "xl"

        pos0 = len(pos_arr[:, 0]) - 1
        i = 0
        if dimsign == "xr":
            max_xx = max(max_x1, max_x2)
            nx = pos_arr[-1, 0]
            while nx < max_xx:
                dt = time_arr[keypoint - i] - time_arr[keypoint - i - 1]
                time_arr = np.append(time_arr, time_arr[pos0 + i] + dt)
                speed_arr = np.append(speed_arr, speed_arr[keypoint - i])
                nx = pos_arr[pos0 + i, 0] + speed_arr[pos0 + i] * dt
                ny = pos_arr[pos0 + i, 1]
                pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
                i += 1
        elif dimsign == "xl":
            min_xx = min(min_x1, min_x2)
            nx = pos_arr[-1, 0]
            while nx > min_xx:
                dt = time_arr[keypoint - i] - time_arr[keypoint - i - 1]
                time_arr = np.append(time_arr, time_arr[pos0 + i] + dt)
                speed_arr = np.append(speed_arr, speed_arr[keypoint - i])
                nx = pos_arr[pos0 + i, 0] - speed_arr[pos0 + i] * dt
                ny = pos_arr[pos0 + i, 1]
                pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
                i += 1
        elif dimsign == "yu":
            max_yy = max(max_y1, max_y2)
            ny = pos_arr[-1, -1]
            while ny < max_yy:
                dt = time_arr[keypoint - i] - time_arr[keypoint - i - 1]
                time_arr = np.append(time_arr, time_arr[pos0 + i] + dt)
                speed_arr = np.append(speed_arr, speed_arr[keypoint - i])
                nx = pos_arr[pos0 + i, 0]
                ny = pos_arr[pos0 + i, 1] + speed_arr[pos0 + i] * dt
                pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
                i += 1
        else:
            min_yy = min(min_y1, min_y2)
            ny = pos_arr[-1, -1]
            while ny > min_yy:
                dt = time_arr[keypoint - i] - time_arr[keypoint - i - 1]
                time_arr = np.append(time_arr, time_arr[pos0 + i] + dt)
                speed_arr = np.append(speed_arr, speed_arr[keypoint - i])
                nx = pos_arr[pos0 + i, 0]
                ny = pos_arr[pos0 + i, 1] - speed_arr[pos0 + i] * dt
                pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
                i += 1
        return pos_arr, speed_arr, time_arr

    def predict_Uturn(pos_arr, speed_arr, time_arr, dimsign, keypoint):
        pos0 = len(pos_arr[:, 0]) - 1
        dtm = pos0 - keypoint
        i = 0
        if dimsign == "x1":
            max_xx = max(max_x1, max_x2)
            nx = pos_arr[-1, 0]
            while nx < max_xx:
                nx = (
                    pos_arr[pos0 + i, 0]
                    + pos_arr[keypoint - dtm - i, 0]
                    - pos_arr[keypoint - dtm - i - 1, 0]
                )
                ny = (
                    pos_arr[pos0 + i, 1]
                    + pos_arr[keypoint - dtm - i, 1]
                    - pos_arr[keypoint - dtm - i - 1, 1]
                )
                pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
                time_arr = np.append(
                    time_arr,
                    time_arr[pos0 + i]
                    + time_arr[keypoint - dtm - i]
                    - time_arr[keypoint - dtm - i - 1],
                )
                speed_arr = np.append(speed_arr, speed_arr[keypoint - dtm - i - 1])
                i += 1
        elif dimsign == "x2":
            min_xx = min(min_x1, min_x2)
            nx = pos_arr[-1, 0]
            while nx > min_xx:
                nx = (
                    pos_arr[pos0 + i, 0]
                    + pos_arr[keypoint - dtm - i, 0]
                    - pos_arr[keypoint - dtm - i - 1, 0]
                )
                ny = (
                    pos_arr[pos0 + i, 1]
                    + pos_arr[keypoint - dtm - i, 1]
                    - pos_arr[keypoint - dtm - i - 1, 1]
                )
                pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
                time_arr = np.append(
                    time_arr,
                    time_arr[pos0 + i]
                    + time_arr[keypoint - dtm - i]
                    - time_arr[keypoint - dtm - i - 1],
                )
                speed_arr = np.append(speed_arr, speed_arr[keypoint - dtm - i - 1])
                i += 1
        elif dimsign == "y1":
            max_yy = max(max_y1, max_y2)
            ny = pos_arr[-1, 1]
            while ny < max_yy:
                nx = (
                    pos_arr[pos0 + i, 0]
                    + pos_arr[keypoint - dtm - i, 0]
                    - pos_arr[keypoint - dtm - i - 1, 0]
                )
                ny = (
                    pos_arr[pos0 + i, 1]
                    + pos_arr[keypoint - dtm - i, 1]
                    - pos_arr[keypoint - dtm - i - 1, 1]
                )
                pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
                time_arr = np.append(
                    time_arr,
                    time_arr[pos0 + i]
                    + time_arr[keypoint - dtm - i]
                    - time_arr[keypoint - dtm - i - 1],
                )
                speed_arr = np.append(speed_arr, speed_arr[keypoint - dtm - i - 1])
                i += 1
        else:
            min_yy = min(min_y1, min_y2)
            ny = pos_arr[-1, 1]
            while ny > min_yy:
                nx = (
                    pos_arr[pos0 + i, 0]
                    + pos_arr[keypoint - dtm - i, 0]
                    - pos_arr[keypoint - dtm - i - 1, 0]
                )
                ny = (
                    pos_arr[pos0 + i, 1]
                    + pos_arr[keypoint - dtm - i, 1]
                    - pos_arr[keypoint - dtm - i - 1, 1]
                )
                pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
                time_arr = np.append(
                    time_arr,
                    time_arr[pos0 + i]
                    + time_arr[keypoint - dtm - i]
                    - time_arr[keypoint - dtm - i - 1],
                )
                speed_arr = np.append(speed_arr, speed_arr[keypoint - dtm - i - 1])
                i += 1
        nx = (
            pos_arr[pos0 + i, 0]
            + pos_arr[keypoint - dtm - i, 0]
            - pos_arr[keypoint - dtm - i - 1, 0]
        )
        ny = (
            pos_arr[pos0 + i, 1]
            + pos_arr[keypoint - dtm - i, 1]
            - pos_arr[keypoint - dtm - i - 1, 1]
        )
        pos_arr = np.append(pos_arr, [[nx, ny]], axis=0)
        time_arr = np.append(
            time_arr,
            time_arr[pos0 + i]
            + time_arr[keypoint - dtm - i]
            - time_arr[keypoint - dtm - i - 1],
        )
        speed_arr = np.append(speed_arr, speed_arr[keypoint - dtm - i - 1])
        return pos_arr, speed_arr, time_arr

    # classify the traj
    def classify(pos_arr, speed_arr, time_arr, dimsign):
        i = len(pos_arr) - 1
        dx = pos_arr[i][0] - pos_arr[i - 1][0]
        dy = pos_arr[i][1] - pos_arr[i - 1][1]
        if dx < 0.1 or dy < 0.1:
            return predict_straight(pos_arr, speed_arr, time_arr, dimsign)
        dflag = dx / dy
        i -= 1
        while True:
            if i < 1:
                break
            dx = pos_arr[i][0] - pos_arr[i - 1][0]
            dy = pos_arr[i][1] - pos_arr[i - 1][1]
            dtmp = dx / dy
            if dx < 0.1 or dy < 0.1:
                return predict_turn(pos_arr, speed_arr, time_arr, dimsign, i)
            if dflag * dtmp < 0:
                return predict_Uturn(pos_arr, speed_arr, time_arr, dimsign, i)
            i -= 1
        return predict_straight(pos_arr, speed_arr, time_arr, dimsign)

    # which traj need to be predicted
    if min_x > max_x:
        if (min_x - pos_arr1[-2, 0]) * (pos_arr1[-1, 0] - pos_arr1[-2, 0]) > 0:
            if min_x - pos_arr1[-2, 0] > 0:
                pos_arr1, speed_arr1, time_arr1 = classify(
                    pos_arr1, speed_arr1, time_arr1, "xr"
                )
            else:
                pos_arr1, speed_arr1, time_arr1 = classify(
                    pos_arr1, speed_arr1, time_arr1, "xl"
                )
        if (min_x - pos_arr2[-2, 0]) * (pos_arr2[-1, 0] - pos_arr2[-2, 0]) > 0:
            if min_x - pos_arr2[-2, 0] > 0:
                pos_arr2, speed_arr2, time_arr2 = classify(
                    pos_arr2, speed_arr2, time_arr2, "xr"
                )
            else:
                pos_arr2, speed_arr2, time_arr2 = classify(
                    pos_arr2, speed_arr2, time_arr2, "xl"
                )
    elif min_y > max_y:
        if (min_y - pos_arr1[-2, 1]) * (pos_arr1[-1, 1] - pos_arr1[-2, 1]) > 0:
            if min_y - pos_arr1[-2, 1] > 0:
                pos_arr1, speed_arr1, time_arr1 = classify(
                    pos_arr1, speed_arr1, time_arr1, "yu"
                )
            else:
                pos_arr1, speed_arr1, time_arr1 = classify(
                    pos_arr1, speed_arr1, time_arr1, "yd"
                )
        if (min_y - pos_arr2[-2, 1]) * (pos_arr2[-1, 1] - pos_arr2[-2, 1]) > 0:
            if min_y - pos_arr2[-2, 1] > 0:
                pos_arr2, speed_arr2, time_arr2 = classify(
                    pos_arr2, speed_arr2, time_arr2, "yu"
                )
            else:
                pos_arr2, speed_arr2, time_arr2 = classify(
                    pos_arr2, speed_arr2, time_arr2, "yd"
                )
    elif (min_xx - pos_arr1[-1, 0]) * (
        pos_arr1[-1, 0] - pos_arr1[-2, 0]
    ) > 0 and math.fabs(pos_arr1[-1, 0] - pos_arr1[-2, 0]) > vehicle_w:
        pos_arr1, speed_arr1, time_arr1 = classify(
            pos_arr1, speed_arr1, time_arr1, "xl"
        )
    elif (min_xx - pos_arr2[-1, 0]) * (
        pos_arr2[-1, 0] - pos_arr2[-2, 0]
    ) > 0 and math.fabs(pos_arr2[-1, 0] - pos_arr2[-2, 0]) > vehicle_w:
        pos_arr2, speed_arr2, time_arr2 = classify(
            pos_arr2, speed_arr2, time_arr2, "xl"
        )
    elif (min_yy - pos_arr1[-1, 1]) * (
        pos_arr1[-1, 1] - pos_arr1[-2, 1]
    ) > 0 and math.fabs(pos_arr1[-1, 1] - pos_arr1[-2, 1]) > vehicle_w:
        pos_arr1, speed_arr1, time_arr1 = classify(
            pos_arr1, speed_arr1, time_arr1, "yd"
        )
    elif (min_yy - pos_arr2[-1, 1]) * (
        pos_arr2[-1, 1] - pos_arr2[-2, 1]
    ) > 0 and math.fabs(pos_arr2[-1, 1] - pos_arr2[-2, 1]) > vehicle_w:
        pos_arr2, speed_arr2, time_arr2 = classify(
            pos_arr2, speed_arr2, time_arr2, "yd"
        )
    elif (max_xx - pos_arr1[-1, 0]) * (
        pos_arr1[-1, 0] - pos_arr1[-2, 0]
    ) > 0 and math.fabs(pos_arr1[-1, 0] - pos_arr1[-2, 0]) > vehicle_w:
        pos_arr1, speed_arr1, time_arr1 = classify(
            pos_arr1, speed_arr1, time_arr1, "xr"
        )
    elif (max_xx - pos_arr2[-1, 0]) * (
        pos_arr2[-1, 0] - pos_arr2[-2, 0]
    ) > 0 and math.fabs(pos_arr2[-1, 0] - pos_arr2[-2, 0]) > vehicle_w:
        pos_arr2, speed_arr2, time_arr2 = classify(
            pos_arr2, speed_arr2, time_arr2, "xr"
        )
    elif (max_yy - pos_arr1[-1, 1]) * (
        pos_arr1[-1, 1] - pos_arr1[-2, 1]
    ) > 0 and math.fabs(pos_arr1[-1, 1] - pos_arr1[-2, 1]) > vehicle_w:
        pos_arr1, speed_arr1, time_arr1 = classify(
            pos_arr1, speed_arr1, time_arr1, "yu"
        )
    elif (max_yy - pos_arr2[-1, 1]) * (
        pos_arr2[-1, 1] - pos_arr2[-2, 1]
    ) > 0 and math.fabs(pos_arr2[-1, 1] - pos_arr2[-2, 1]) > vehicle_w:
        pos_arr2, speed_arr2, time_arr2 = classify(
            pos_arr2, speed_arr2, time_arr2, "yu"
        )
    return pos_arr1, pos_arr2, speed_arr1, speed_arr2, time_arr1, time_arr2


def eval_diversity(pos_ego, pos_agent, speed_ego, speed_agent, time_ego, time_agent):
    # count intersection point
    pos_list1, pos_list2 = count_intersection_area(pos_ego, pos_agent)
    while True:
        if (len(pos_list1) > eps) and (len(pos_list2) > eps):
            break
        pos_ego, pos_agent, speed_ego, speed_agent, time_ego, time_agent = predict_traj(
            pos_ego, pos_agent, speed_ego, speed_agent, time_ego, time_agent
        )
        pos_list1, pos_list2 = count_intersection_area(pos_ego, pos_agent)
    A1, B1, A2, B2, P1, P2 = count_intersection_segment(pos_list1, pos_list2)
    # count points index
    idx_A1, idx_B1 = (
        pos_ego.tolist().index([A1.x, A1.y]),
        pos_ego.tolist().index([B1.x, B1.y]),
    )
    assert (
        (idx_A1 > 0) and (idx_B1 < pos_ego.shape[0] - 1) and (idx_A1 < idx_B1)
    ), "count intersection area: wrong idx1"
    idx_A2, idx_B2 = (
        pos_agent.tolist().index([A2.x, A2.y]),
        pos_agent.tolist().index([B2.x, B2.y]),
    )
    assert (
        (idx_A2 > 0) and (idx_B2 < pos_agent.shape[0] - 1) and (idx_A2 < idx_B2)
    ), "count intersection area: wrong idx2"

    # -----------use straight cross------------#
    # count intersection point
    # P = CountSegIntersect(A1, B1, A2, B2)
    t1 = Count_dtimespeed(
        speed_ego[idx_A1],
        speed_ego[idx_B1],
        CountPointDist(A1, P1),
        CountPointDist(A1, B1),
    )
    t2 = Count_dtimespeed(
        speed_agent[idx_A2],
        speed_agent[idx_B2],
        CountPointDist(A2, P2),
        CountPointDist(A2, B2),
    )

    first_agent, arrive_time, time_diff = count_time_score(
        time_ego, time_agent, idx_A1, idx_A2, t1, t2
    )
    time_diff = np.clip(time_diff, -8, 8)
    time_score = time_diff / max_time_diff
    # count dist_score
    if first_agent == 1:
        Q, dist_diff = count_dist_score(pos_agent, time_agent, arrive_time, P2)
    else:
        Q, dist_diff = count_dist_score(pos_ego, time_ego, arrive_time, P1)
        dist_diff = -dist_diff
    dist_diff = np.clip(dist_diff, -30, 30)
    dist_score = dist_diff / max_dist_diff
    return time_score, dist_score

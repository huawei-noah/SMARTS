import bisect
import math

import numpy as np


class CubicSpline1D:
    """
    1D Cubic Spline class
    Parameters
    ----------
    x : list
        x coordinates for data points. This x coordinates must be
        sorted
        in ascending order.
    y : list
        y coordinates for data points
    """

    def __init__(self, x, y):
        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted in ascending order")

        self.a, self.b, self.c, self.d = [], [], [], []
        self.x = x
        self.y = y
        self.nx = len(x)  # dimension of x

        # calc coefficient a
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h, self.a)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            d = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            b = 1.0 / h[i] * (self.a[i + 1] - self.a[i]) - h[i] / 3.0 * (
                2.0 * self.c[i] + self.c[i + 1]
            )
            self.d.append(d)
            self.b.append(b)

    def calc_position(self, x):
        """
        Calc `y` position for given `x`.
        if `x` is outside the data point's `x` range, return None.
        Returns
        -------
        y : float
            y position for given x.
        """
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        position = (
            self.a[i] + self.b[i] * dx + self.c[i] * dx**2.0 + self.d[i] * dx**3.0
        )

        return position

    def calc_first_derivative(self, x):
        """
        Calc first derivative at given x.
        if x is outside the input x, return None
        Returns
        -------
        dy : float
            first derivative for given x.
        """

        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        dy = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx**2.0

        return dy

    def calc_second_derivative(self, x):
        """
        Calc second derivative at given x.
        if x is outside the input x, return None
        Returns
        -------
        ddy : float
            second derivative for given x.
        """

        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        ddy = 2.0 * self.c[i] + 6.0 * self.d[i] * dx

        return ddy

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0

        return A

    def __calc_B(self, h, a):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = (
                3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] - 3.0 * (a[i + 1] - a[i]) / h[i]
            )

        return B


class CubicSpline2D:
    """
    Cubic CubicSpline2D class
    Parameters
    ----------
    x : list
        x coordinates for data points.
    y : list
        y coordinates for data points.
    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))

        return s

    def calc_position(self, s):
        """
        calc position
        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.
        Returns
        -------
        x : float
            x position for given s.
        y : float
            y position for given s.
        """
        x = self.sx.calc_position(s)
        y = self.sy.calc_position(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.
        Returns
        -------
        k : float
            curvature for given s.
        """
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        k = (ddy * dx - ddx * dy) / ((dx**2 + dy**2) ** (3 / 2))

        return k

    def calc_yaw(self, s):
        """
        calc yaw
        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.
        Returns
        -------
        yaw : float
            yaw angle (tangent vector) for given s.
        """
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        yaw = math.atan2(dy, dx)

        return yaw


class FrenetPath:
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


def generate_target_course(x, y):
    csp = CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        try:
            ix, iy = csp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(csp.calc_yaw(i_s))
            rk.append(csp.calc_curvature(i_s))
        except:
            continue

    return rx, ry, ryaw, rk, csp


class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array(
            [
                [time**3, time**4, time**5],
                [3 * time**2, 4 * time**3, 5 * time**4],
                [6 * time, 12 * time**2, 20 * time**3],
            ]
        )
        b = np.array(
            [
                xe - self.a0 - self.a1 * time - self.a2 * time**2,
                vxe - self.a1 - 2 * self.a2 * time,
                axe - 2 * self.a2,
            ]
        )
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = (
            self.a0
            + self.a1 * t
            + self.a2 * t**2
            + self.a3 * t**3
            + self.a4 * t**4
            + self.a5 * t**5
        )

        return xt

    def calc_first_derivative(self, t):
        xt = (
            self.a1
            + 2 * self.a2 * t
            + 3 * self.a3 * t**2
            + 4 * self.a4 * t**3
            + 5 * self.a5 * t**4
        )

        return xt

    def calc_second_derivative(self, t):
        xt = (
            2 * self.a2
            + 6 * self.a3 * t
            + 12 * self.a4 * t**2
            + 20 * self.a5 * t**3
        )

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

        return xt


class CubicPolynomial:
    def __init__(self, vxs, axs, vxe, axe, time):
        self.a0 = vxs
        self.a1 = axs

        A = np.array([[time**2, time**3], [2 * time, 3 * time**2]])
        b = np.array([vxe - self.a0 - self.a1 * time, axe - self.a1])
        x = np.linalg.solve(A, b)

        self.a2 = x[0]
        self.a3 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t

        return xt


def generate_lon_profile(v_s, a_s, acc):
    v_target = np.clip(v_s + acc * 3, 0, 16)
    if acc != 0:
        t_target = round((v_target - v_s) / acc, 3)
        t_target = np.clip(t_target, 0.1, 3)
    else:
        t_target = 3

    T = np.arange(0, t_target, 0.1)
    lon_profile = CubicPolynomial(v_s, a_s, v_target, 0, t_target)
    speed = [lon_profile.calc_point(t) for t in T]

    if len(speed) < 30:
        speed.extend([speed[-1] for _ in range(30 - len(speed))])

    speed = np.clip(speed, 0.01, 16)
    displacement = np.cumsum(speed * 0.1)

    return speed, displacement


def generate_lat_profile(d, v_d):
    d_target = 0
    t_target = np.clip(np.abs(d - d_target) / 1.5, 0.1, 3)
    T = np.arange(0.1, t_target + 0.1, 0.1)
    lat_profile = QuinticPolynomial(d, v_d, 0, d_target, 0, 0, t_target)
    displacement = [lat_profile.calc_point(t) for t in T]
    speed = [lat_profile.calc_first_derivative(t) for t in T]

    if len(speed) < 30:
        speed.extend([speed[-1] for _ in range(30 - len(speed))])
        displacement.extend([displacement[-1] for _ in range(30 - len(displacement))])

    return speed, displacement

from time import perf_counter
import numpy as np


def line_intersect1(a, b, c, d):
    """Check if the lines [a, b] and [c, d] intersect, and return the
    intersection point if so. Otherwise, return None.
      d
    a─┼─b
      c
    """

    r = b - a
    s = d - c
    d = r[0] * s[1] - r[1] * s[0]

    if d == 0:
        return None

    u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / d
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / d

    if 0 <= u <= 1 and 0 <= t <= 1:
        return a + t * r

    return None


def line_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Check if the lines [a, b] and [c, d] intersect, and return the
    intersection point if so. Otherwise, return None.
      d
    a─┼─b
      c
    """

    r = b - a
    s = d - c
    d = r[0] * s[1] - r[1] * s[0]

    if d == 0:
        return False

    u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / d
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / d

    if 0 <= u <= 1 and 0 <= t <= 1:
        return True
    return False


def line_intersect_vectorized(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> bool:
    r = b - a
    s = d - c
    rs1 = np.multiply(r[:, 0], s[:, 1])
    rs2 = np.multiply(r[:, 1], s[:, 0])
    d = rs1 - rs2

    if not np.any(d):
        return False

    u = np.divide(
        np.multiply(c[:, 0] - a[:, 0], r[:, 1])
        - np.multiply(c[:, 1] - a[:, 1], r[:, 0]),
        d,
    )
    t = np.divide(
        np.multiply(c[:, 0] - a[:, 0], s[:, 1])
        - np.multiply(c[:, 1] - a[:, 1], s[:, 0]),
        d,
    )

    if np.any((0 <= u) & (u <= 1)) and np.any((0 <= t) & (t <= 1)):
        return True
    return False


def intersect_loop(line1: np.ndarray, line2: np.ndarray) -> bool:
    for i in range(len(line1) - 1):
        a = line1[i]
        b = line1[i + 1]
        for j in range(len(line2) - 1):
            c = line2[j]
            d = line2[j + 1]
            if line_intersect(a, b, c, d):
                return True
    return False


def intersect_vectorized(line1: np.ndarray, line2: np.ndarray) -> bool:
    C = np.roll(line2, 0)[:-1]
    D = np.roll(line2, -1, axis=0)[:-1]
    for i in range(len(line1) - 1):
        A = np.tile(line1[i], (len(C), 1))
        B = np.tile(line1[i + 1], (len(C), 1))
        if line_intersect_vectorized(A, B, C, D):
            return True
    return False


def main():
    line1 = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
        ]
    )
    line2 = np.array(
        [
            [3, 0],
            [2, 1],
            [1, 2],
            [0, 3],
        ]
    )
    line3 = np.array(
        [
            [1, 0],
            [2, 1],
            [3, 2],
            [4, 3],
        ]
    )
    loop_result = intersect_loop(line1, line2)
    vectorized_result = intersect_vectorized(line1, line2)
    assert loop_result == vectorized_result

    loop_result = intersect_loop(line1, line3)
    vectorized_result = intersect_vectorized(line1, line3)
    assert loop_result == vectorized_result

    n = 10000
    start = perf_counter()
    for _ in range(n):
        line1 = np.random.uniform(0, 100, (20, 2))
        line2 = np.random.uniform(100, 200, (20, 2))
        loop_result = intersect_loop(line1, line2)
    end = perf_counter()
    elapsed_time = round((end - start), 3)
    print(f"Loop: {elapsed_time} s")

    start = perf_counter()
    for _ in range(n):
        line1 = np.random.uniform(0, 100, (20, 2))
        line2 = np.random.uniform(100, 200, (20, 2))
        vectorized_result = intersect_vectorized(line1, line2)
    end = perf_counter()
    elapsed_time = round((end - start), 3)
    print(f"Vectorized: {elapsed_time} s")


if __name__ == "__main__":
    main()

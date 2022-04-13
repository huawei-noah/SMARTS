import numpy as np

from smarts.core.utils.math import position_to_ego_frame, world_position_from_ego_frame


def test_ego_centric_conversion():
    p_start = [1, 2, 3]
    pe = [1, -5, 2]
    he = -3

    pec = position_to_ego_frame(p_start, pe, he)

    assert np.allclose([-0.9878400564190705, -6.929947476203118, 1.0], pec)

    p_end = world_position_from_ego_frame(pec, pe, he)

    assert np.allclose(p_end, p_start)

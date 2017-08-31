import numpy as np
import quantities as pq
import pytest


def test_sign_change():
    from edog.tools import _find_sign_change

    a = np.array([1, 1, -1, -2, -3, 4, 5])
    a_sign = np.array([0, 0, 1, 0, 0, 1, 0])
    assert((a_sign == _find_sign_change(a)).all())

    a = np.array([-1, 0, 1])
    a_sign = np.array([0, 1, 1])
    assert((a_sign == _find_sign_change(a)).all())

    a = np.array([-1, 0, -1, 1, 2])
    a_sign = np.array([0, 1, 1, 1, 0])
    assert((a_sign == _find_sign_change(a)).all())

"""Pruebas del cálculo de valores esperados."""

from src.probability import calculate_expected_value


def test_calculation_with_typical_values():
    probability = 0.004
    benefit = 500_000

    expected = 2_000
    actual = calculate_expected_value(probability, benefit)

    assert actual == expected


def test_zero_probability_produces_zero():
    assert calculate_expected_value(0, 500_000) == 0


def test_certain_event_returns_the_full_benefit():
    assert calculate_expected_value(1, 750) == 750


def test_negative_probability_is_rejected():
    try:
        calculate_expected_value(-0.1, 500)
        assert False
    except ValueError:
        pass


def test_probability_above_one_is_rejected():
    try:
        calculate_expected_value(1.1, 500)
        assert False
    except ValueError:
        pass

"""Money has to be exact, so these tests are about the awkward cases.

Most of them exist because a float implementation passes the easy ones and
fails here, which is precisely why the money type is integer paise.
"""

from decimal import Decimal

import pytest

from stockmate.money import Money, parse_money, quantize_half_up, total


def test_rupees_round_trip():
    assert Money.from_rupees("1234.56").paise == 123456
    assert Money.from_rupees("1234.56").rupees == Decimal("1234.56")
    assert str(Money.from_rupees("1234.56")) == "1,234.56"


def test_float_input_uses_shortest_repr():
    # 18.07 is not representable in binary; a naive int(x * 100) gives 1806.
    assert Money.from_rupees(18.07).paise == 1807
    assert Money.from_rupees(0.29).paise == 29
    assert Money.from_rupees(1.005).paise == 101


def test_half_up_not_bankers_rounding():
    # Python's round() would give 2 here, which is not what a bill does.
    assert quantize_half_up(Decimal("2.5")) == Decimal("3")
    assert quantize_half_up(Decimal("3.5")) == Decimal("4")


def test_addition_is_exact_over_many_terms():
    # The classic float failure: 0.1 added a hundred times is not 10.
    cents = [Money.from_rupees("0.10") for _ in range(100)]
    assert total(cents) == Money.from_rupees("10.00")


def test_multiplication_rounds_once():
    assert (Money.from_rupees("33.33") * Decimal("3")).paise == 9999
    assert (Money.from_rupees("10.00") * Decimal("0.335")).paise == 335


def test_percent():
    assert Money.from_rupees("1000.00").percent(18).paise == 18000
    assert Money.from_rupees("99.99").percent(5).paise == 500


def test_split_half_preserves_the_total():
    # CGST and SGST must add back to the tax that was charged, odd paise and all.
    for paise in (0, 1, 2, 3, 101, 12345, 99999):
        a, b = Money(paise).split_half()
        assert a + b == Money(paise)
        assert abs(a.paise - b.paise) <= 1


def test_cannot_add_a_bare_number():
    with pytest.raises(TypeError):
        Money.from_rupees("1.00") + 1


@pytest.mark.parametrize(
    "text,paise",
    [
        ("Rs 1,234.50", 123450),
        ("₹900/-", 90000),
        ("18,000", 1800000),
        ("1234", 123400),
        ("45.5", 4550),
        ("INR 2,00,000.00", 20000000),
    ],
)
def test_parse_money_reads_receipt_text(text, paise):
    parsed = parse_money(text)
    assert parsed is not None
    assert parsed.paise == paise


def test_parse_money_returns_none_not_zero():
    # A missing amount must never silently become zero, because zero is a
    # perfectly plausible total and would be checked as if it were real.
    assert parse_money("") is None
    assert parse_money("TOTAL") is None
    assert parse_money(None) is None

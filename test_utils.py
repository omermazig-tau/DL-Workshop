import pytest

from utils import add_seconds_to_time


@pytest.mark.parametrize("text_time_input,text_time_after_increase_input", [
    ("3:45", "3:49"),
    ("9:57", "10:01"),
    ("0:58", "1:02"),
    ("0:52", "56."),
])
def test_add_time(text_time_input, text_time_after_increase_input):
    result_text_time = add_seconds_to_time(text_time_input, 4)
    assert result_text_time == text_time_after_increase_input


@pytest.mark.parametrize("text_time_input,text_time_after_decrease_input", [
    ("3:49", "3:45"),
    ("10:01", "9:57"),
    ("1:02", "58."),
    ("0:56", "52."),
    ("0:06", "2."),
    ("0:02", "0."),
])
def test_decrease_time(text_time_input, text_time_after_decrease_input):
    result_text_time = add_seconds_to_time(text_time_input, -4)
    assert result_text_time == text_time_after_decrease_input

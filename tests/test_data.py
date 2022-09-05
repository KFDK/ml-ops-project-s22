from contextlib import contextmanager
from src.data.make_dataset import my_tokenize
import pytest


@contextmanager
def does_not_raise():
    yield


def test_tokenizer_input_ids():
    x = "this is a test, my g"
    x = my_tokenize(x)["input_ids"]
    for input_id in x:
        assert isinstance(input_id, int)


def test_attention_mask_lenght():
    x = "this is a test, my g"
    x = my_tokenize(x)["attention_mask"]
    for attention_mask in x:
        assert isinstance(attention_mask, int)


@pytest.mark.parametrize(
    "test_input, expectation",
    [
        ("", pytest.raises(ValueError)),
        ("  ", pytest.raises(ValueError)),
        ([], pytest.raises(ValueError)),
        ("this is a test, my g", does_not_raise()),
    ],
)
def test_empty_inputs_to_tokenizer(test_input, expectation):
    with expectation:
        my_tokenize(test_input)

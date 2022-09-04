from src.data.make_dataset import my_tokenize

def test_tokenizer():
    x = 'this is a test, my g'
    x=my_tokenize(x)["input_ids"][0]
    assert(not type(x)==str)
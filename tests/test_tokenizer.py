from scratchlm.tokenizer import BPETokenizer, WordTokenizer, basic_tokenize

CORPUS = "the quick brown fox the quick brown dog lower lowest newer newest the lazy fox"


def test_basic_tokenize():
    assert basic_tokenize("Hello, world!") == ["Hello", ",", "world", "!"]
    assert basic_tokenize("Hello, World!", lower=True) == ["hello", ",", "world", "!"]


def test_bpe_round_trip():
    tok = BPETokenizer(vocab_size=120)
    tok.train(CORPUS)
    text = "the quick fox"
    assert tok.decode(tok.encode(text)) == text


def test_bpe_specials_and_size():
    tok = BPETokenizer(vocab_size=120)
    tok.train(CORPUS)
    for special in ["<pad>", "<unk>", "<s>", "</s>"]:
        assert special in tok.vocab
    assert tok.eos_token_id == tok.vocab["</s>"]
    assert tok.vocab_size == len(tok.vocab)


def test_bpe_is_deterministic():
    a, b = BPETokenizer(vocab_size=120), BPETokenizer(vocab_size=120)
    a.train(CORPUS)
    b.train(CORPUS)
    assert a.merges == b.merges
    assert a.vocab == b.vocab


def test_bpe_serialization_round_trip():
    tok = BPETokenizer(vocab_size=120)
    tok.train(CORPUS)
    restored = BPETokenizer.from_dict(tok.to_dict())
    assert restored.encode("the quick fox") == tok.encode("the quick fox")


def test_word_tokenizer_round_trip():
    tok = WordTokenizer()
    tok.train(CORPUS)
    assert tok.decode(tok.encode("the quick fox")) == "the quick fox"

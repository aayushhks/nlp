# CS505: NLP - Spring 2026

from nltk.tokenize import word_tokenize
import collections
import re

class WordTokenizer:
    """
    A simple baseline tokenizer that splits on whitespace and handles unknown tokens.
    Use this if you get stuck on the BPE implementation and decide to implement
    the language models first.
    """
    def __init__(self, vocab_size=None):
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
        self.vocab_size = vocab_size

    def train(self, text):
        # 1. Count word frequencies
        word_counts = collections.Counter(word_tokenize(text))
        
        # 2. Determine vocabulary size
        # If vocab_size is not set, we use all unique words + specials
        num_words_to_keep = len(word_counts)
        if self.vocab_size is not None:
            num_words_to_keep = self.vocab_size - len(self.special_tokens)

        # 3. Get most common words
        most_common = word_counts.most_common(num_words_to_keep)
        
        # 4. Initialize vocab with special tokens
        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
            
        # 5. Add common words to vocab
        current_id = len(self.vocab)
        for word, _ in most_common:
            if word not in self.vocab:
                self.vocab[word] = current_id
                self.inverse_vocab[current_id] = word
                current_id += 1
                
        print(f"WordTokenizer training complete. Vocab size: {len(self.vocab)}")

    def tokenize(self, text):
        """
        Converts text to IDs. Words not in vocab are mapped to <unk>.
        """
        unk_id = self.vocab["<unk>"]
        tokens = []
        for word in word_tokenize(text):
            tokens.append(self.vocab.get(word, unk_id))
        return tokens


class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = {}  # Map token -> id
        self.inverse_vocab = {} # Map id -> token

        # merges learned in order: list of ((a,b), "ab")
        self.merges = []
        # rank lookup: (a,b) -> rank (lower = earlier = higher priority)
        self.bpe_ranks = {}

        self.special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]

    def get_stats(self, vocab_counts):
        # TODO: Count frequency of all token pairs in the current vocabulary.
        # This method should return a dictionary with the following structure:
        # {[token1 (str), token2 (str)]: frequency (int)}
        # STUDENT START ------------------------------------
        pass
        # STUDENT END --------------------------------------

    def merge_vocab(self, pair, v_in):
        """
        Merge the most frequent pair in the vocabulary.
        """
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def train(self, text):
        # TODO: 1. Pre-tokenize text into words and characters.
        # We add ▁ (Note: this is not an underscore! It's U+2581.) to
        # mark end of words to handle suffixes correctly.
        # STUDENT START ----------------------------
        # STUDENT END ------------------------------

        # TODO: 2. Iteratively merge the most frequent token pairs.
        # You will need to (i) compute the frequency of each pair,
        # (ii) find the most frequent pair, (iii) store the merge in
        # `self.merges`, and (iv) update the counts using `self.merge_vocab`.
        # HINT: You might be tempted to recompute all the counts from scratch every
        # time you do a merge, but there is a more efficient way to recompute counts.
        # Merge until we reach vocab_size (including special tokens).
        # Stop early if there are no pairs left, or if we reach the
        # target vocab size.
        # STUDENT START ---------------------------------------------
        # STUDENT END -----------------------------------------------

        # TODO: 3. Update `self.vocab` and `self.inverse_vocab`.
        # We start with the special tokens:
        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token

        # Then, we add the learned tokens and the characters.
        # HINT: if you stored all of the token pairs above, you could
        # repurpose that to build your vocabulary indices.
        # STUDENT START ---------------------------------------------
        # STUDENT END -------------------------------------------------

        print(f"Training complete. Vocab size: {len(self.vocab)}")

    def _apply_bpe(self, symbols):
        """
        Apply BPE merges to a list of symbols, in correct priority order.
        symbols: List[str]
        returns: List[str]
        """
        # keep merging until no mergeable pairs exist
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            # pick best ranked pair among those present
            ranked = [(self.bpe_ranks[p], p) for p in pairs if p in self.bpe_ranks]
            if not ranked:
                break

            _, best_pair = min(ranked)  # smaller rank = earlier learned

            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    new_symbols.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1

            symbols = new_symbols

        return symbols

    def tokenize(self, text):
        tokens = []

        for word in word_tokenize(text):
            # same pre-tokenization scheme as training
            symbols = list(word) + ["▁"]

            # apply merges in order
            symbols = self._apply_bpe(symbols)

            tokens.extend(symbols)

        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]
        return ids
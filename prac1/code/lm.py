import argparse
import math
import time
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tokenizer import BPETokenizer, WordTokenizer  # You can use WordTokenizer if you don't get around
                                    # to implementing BPETokenizer.

def train_neural_model(model, train_data, vocab_size, epochs=2, batch_size=32, lr=0.001, device='cpu'):
    # Create dataset (Batch, Seq_Len)
    # Simplified data loader creation
    seq_len = 30
    x_list, y_list = [], []
    for i in range(0, len(train_data) - seq_len, seq_len):
        chunk = train_data[i:i+seq_len+1]
        if len(chunk) < seq_len + 1:
            continue
        x_list.append(chunk[:-1])
        y_list.append(chunk[1:]) # Next token prediction
    
    X = torch.tensor(x_list, dtype=torch.long).to(device)
    Y = torch.tensor(y_list, dtype=torch.long).to(device)
    
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.NLLLoss() # Since models return log_softmax
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        for bx, by in loader:
            optimizer.zero_grad()
            log_probs, _ = model(bx) # Output: (Batch, Seq, Vocab)
            
            # Flatten for loss
            loss = criterion(log_probs.view(-1, vocab_size), by.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f} | Time: {time.time()-start_time:.2f}s")


class NGramLM:
    def __init__(self, n, k=1.0):
        self.n = n
        self.k = k
        self.ngram_counts = collections.defaultdict(int)
        self.context_counts = collections.defaultdict(int)
        self.vocab = set()

    def get_ngrams(self, tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def train(self, tokens):
        # TODO: Loop through all the ngrams in `self.get_ngrams`.
        # Store the count of all bigram prefix counts
        # in `self.context_counts`, and store the count of
        # all trigrams in `self.ngram_counts`.
        # STUDENT START --------------------------
        pass
        # STUDENT END -----------------------------
            
    def get_prob(self, context, token):
        # TODO: Calculate P(token | context) with Laplace smoothing.
        # Remember to use `self.k`!
        # STUDENT START ------------------------------------
        pass
        # STUDENT END ---------------------------------------

    def perplexity(self, test_tokens):
        # TODO: compute perplexity on the test set.
        # STUDENT START --------------------------------
        pass
        # STUDENT END -----------------------------------

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size=100, hidden_size=100):
        super(RNNLM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # RNN Parameters: h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
        self.W_ih = nn.Parameter(torch.Tensor(embed_size, hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_size))
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def rnn_cell(self, x, h_prev):
        # The RNN forward pass is defined as: h_t = tanh(x_t @ W_ih + b_ih + h_{t-1} @ W_hh + b_hh)
        h_t = torch.tanh(x @ self.W_ih + self.b_ih + h_prev @ self.W_hh + self.b_hh)
        return h_t

    def forward(self, x, hidden=None):
        batch_size, seq_len = x.size()
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size).to(x.device)
        outputs = []

        # TODO: you will first embed x using self.embedding.
        # STUDENT START --------------------------------------
        # STUDENT END ----------------------------------------
        
        # TODO: now, you will loop over all indices in seq_len, and
        # compute the hidden state h_i for each. This involves running
        # the self.rnn_cell over the embeddings for a particular position.
        # Append this hidden state to `outputs`.
        # STUDENT START --------------------------------
        # STUDENT END ----------------------------------
        
        # TODO: Concatenate the outputs into a tensor. Then,
        # use the output layer `self.fc` to compute the logits
        # over all possible tokens in the vocabulary.
        # STUDENT START -----------------------------------
        # STUDENT END -------------------------------------

    def get_perplexity(self, data_loader, device='cpu'):
        self.eval()
        total_nll = 0
        total_tokens = 0
        criterion = nn.NLLLoss(reduction='sum')
        
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                log_probs, _ = self(x)
                # Reshape for loss calculation
                loss = criterion(log_probs.view(-1, self.vocab_size), y.view(-1))
                total_nll += loss.item()
                total_tokens += y.numel()
                
        return math.exp(total_nll / total_tokens)


# Extra credit!
class LSTMLM(RNNLM):
    def __init__(self, vocab_size, embed_size=100, hidden_size=100):
        # Initialize RNNLM to get embedding and fc layers
        super(LSTMLM, self).__init__(vocab_size, embed_size, hidden_size)
        
        # LSTM Parameters: 4 gates (i, f, g, o)
        # We use a single large matrix for efficiency: (input_size, 4 * hidden_size)
        self.W_ih_lstm = nn.Parameter(torch.Tensor(embed_size, 4 * hidden_size))
        self.W_hh_lstm = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.b_ih_lstm = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_hh_lstm = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.init_weights()

    def lstm_cell(self, x, h_prev, c_prev):
        # TODO: implement the LSTM cell. This method takes as input the
        # h_{i-1}, c_{i-1}, and the input x. It returns h_i and c_i.
        # STUDENT START ------------------------------------------
        pass
        # STUDENT END ---------------------------------------------

    def forward(self, x, states=None):
        batch_size, seq_len = x.size()
        if states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = states
        
        # TODO: implement the rest of the forward pass. This looks a lot like
        # the RNN implementation, but using `self.lstm_cell` instead of `self.rnn_cell`.
        # One key difference is that `self.lstm_cell` expects different arguments,
        # and thus requires you to track the hidden state h_t and the
        # context vector c_t.
        # STUDENT START --------------------------------------
        # STUDENT END ----------------------------------------
    
    # NOTE: you do not need to implement perplexity again. The logic is exactly the same
    # as it was for the RNNLM, which this class inherits.



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["NGRAM", "RNN", "LSTM"])
    args = parser.parse_args()

    print(f"Running {args.model} Language Model...")

    # Load train / dev / test data
    print("Loading data...")
    with open("../data/train.txt", "r", encoding="utf-8") as f:
        train_text = f.read()
    with open("../data/dev.txt", "r", encoding="utf-8") as f:
        dev_text = f.read()
    with open("../data/test.txt", "r", encoding="utf-8") as f:
        test_text = f.read()

    # Tokenization (train ONLY)
    start_train = time.time()

    print("Tokenizing data...")
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train(train_text)

    # TODO: For Q2, tokenize a few words manually. You
    # can use `tokenizer.tokenize()` for this.
    # STUDENT START -----------------------
    # STUDENT END -------------------------

    train_data = tokenizer.tokenize(train_text)
    dev_data   = tokenizer.tokenize(dev_text)
    test_data  = tokenizer.tokenize(test_text)

    vocab_size = len(tokenizer.vocab)
    print(f"Vocab Size: {vocab_size}")

    # N-gram Models
    if args.model == "NGRAM":
        lm = NGramLM(n=3, k=1.0)
        lm.train(train_data)

        train_time = time.time() - start_train

        start_eval = time.time()
        dev_ppl = lm.perplexity(dev_data)
        test_ppl = lm.perplexity(test_data)

        print(f"Training Time: {train_time:.2f}s")
        print(f"Dev Perplexity:  {dev_ppl:.4f}")
        print(f"Test Perplexity: {test_ppl:.4f}")

    # Neural Models
    elif args.model in ["RNN", "LSTM"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def make_loader(data):
            seq_len = 30
            x, y = [], []
            for i in range(0, len(data) - seq_len, seq_len):
                chunk = data[i:i + seq_len + 1]
                if len(chunk) < seq_len + 1:
                    continue
                x.append(chunk[:-1])
                y.append(chunk[1:])
            return DataLoader(
                TensorDataset(torch.tensor(x), torch.tensor(y)),
                batch_size=32,
                shuffle=False
            )

        dev_loader  = make_loader(dev_data)
        test_loader = make_loader(test_data)

        if args.model == "RNN":
            model = RNNLM(vocab_size).to(device)
            train_neural_model(model, train_data, vocab_size, device=device)

        elif args.model == "LSTM":
            model = LSTMLM(vocab_size).to(device)
            train_neural_model(model, train_data, vocab_size, device=device)

        train_time = time.time() - start_train

        start_eval = time.time()
        dev_ppl = model.get_perplexity(dev_loader, device=device)
        test_ppl = model.get_perplexity(test_loader, device=device)

        print(f"Training Time: {train_time:.2f}s")
        print(f"Dev Perplexity:  {dev_ppl:.4f}")
        print(f"Test Perplexity: {test_ppl:.4f}")

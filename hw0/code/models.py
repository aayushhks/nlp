# CS505: NLP - Spring 2026
import torch
import numpy as np
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
import random

class BoWFeaturizer:
    """
    This is a bag-of-words featurizer. It uses `build_vocab` to load a list of Examples
    and uses the top `max_vocab_size` words by frequency as its vocabulary.
    For a given Example, it counts the number of instances of each word in
    `self.vocab` and returns this vector of counts.
    """
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.vocab = {} # mapping word -> index
        self.inverse_vocab = {}
        self.vocab_size = 0

    def build_vocab(self, data):
        counts = Counter()
        # TODO: count the number instances of each token (here, just words and
        # punctuation) in `data`. Filter the vocab down to the `self.max_vocab_size` most
        # frequent tokens, and put these in a variable called `most_common`. HINT:
        # you can use the `word_tokenize` function that's been
        # imported above to tokenize the string.
        # STUDENT START ---------------------------------
        for ex in data:
            # [cite_start]Lowercase for better normalization [cite: 212]
            tokens = word_tokenize(ex.text.lower())
            counts.update(tokens)

            # [cite_start]Filter to max_vocab_size [cite: 207]
        most_common = counts.most_common(self.max_vocab_size)
        # STUDENT END ------------------------------------
        
        # you might need to remove the `count` variable here, depending on how you
        # implemented the above.
        self.vocab = {word: idx for idx, (word, count) in enumerate(most_common)}
        self.inverse_vocab = {idx: word for idx, (word, count) in enumerate(most_common)}
        self.vocab_size = len(self.vocab)

        print(f"Vocabulary built with {self.vocab_size} words.")

    def get_feature_vector(self, text):
        # TODO: Return a bag-of-words feature vector. Each index in
        # the vocabulary should have a corresponding index in this vector.
        # A token's vector index should contain the frequency of that token
        # in `text`.
        # This shold return a torch tensor of size (vocab_size,).
        # STUDENT START -------------------------
        # [cite_start]Create zero vector [cite: 215]
        vec = torch.zeros(self.vocab_size)
        tokens = word_tokenize(text.lower())

        for token in tokens:
            if token in self.vocab:
                vec[self.vocab[token]] += 1
        return vec
        # STUDENT END ---------------------------


class BigramFeaturizer(BoWFeaturizer):
    def build_vocab(self, data):
        counts = Counter()
        for ex in data:
            tokens = word_tokenize(ex.text.lower())
            # TODO: generate bigrams
            # STUDENT START ----------------------------
            # [cite_start]Generate bigrams [cite: 258]
            bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
            counts.update(bigrams)
            # STUDENT END ------------------------------
        
        # TODO: build your vocabulary of the `self.max_vocab_size` most frequent bigrams.
        # STUDENT START -------------------------------------------
        most_common = counts.most_common(self.max_vocab_size)
        self.vocab = {word: idx for idx, (word, count) in enumerate(most_common)}
        self.inverse_vocab = {idx: word for idx, (word, count) in enumerate(most_common)}
        self.vocab_size = len(self.vocab)
        print(f"Bigram Vocabulary built with {self.vocab_size} features.")
        # STUDENT END ---------------------------------------------

    def get_feature_vector(self, text):
        tokens = word_tokenize(text.lower())
        vec = torch.zeros(self.vocab_size)
        
        # TODO: use the list of tokens to generate bigram features.
        # Return the bigram feature vector.
        # STUDENT START --------------------------------------
        bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]

        for bigram in bigrams:
            if bigram in self.vocab:
                vec[self.vocab[bigram]] += 1
        return vec
        # STUDENT END -----------------------------------------


class CustomFeaturizer(BoWFeaturizer):
    """
    Q7: Custom Featurizer.
    Extends BoW with two additional features:
    1. Log(Length of text)
    2. Number of numeric digits in text
    """

    def build_vocab(self, data):
        super().build_vocab(data)
        # We add 2 extra dimensions for our custom features
        self.vocab_size += 2
        print(f"Custom Vocabulary built. Base: {self.vocab_size - 2}, Total: {self.vocab_size}")

    def get_feature_vector(self, text):
        # 1. Get base vector.
        # Since self.vocab_size is 5002, this returns a vector of size 5002.
        # The indices 0-4999 are filled with word counts. Indices 5000-5001 are 0.
        vec = super().get_feature_vector(text)

        # 2. Calculate custom features
        tokens = word_tokenize(text.lower())
        feat_len = torch.log(torch.tensor(len(tokens) + 1.0))
        feat_digits = torch.tensor(sum(c.isdigit() for c in text) * 1.0)

        # 3. Assign to the last 2 reserved indices instead of concatenating
        vec[-2] = feat_len
        vec[-1] = feat_digits

        return vec

class BlackBoxClassifier(torch.nn.Module):
    """
    This is a logistic regression classifier using PyTorch's built-in modules.
    Only used in Task 1. You will implement something like this from scratch
    in the LogisticRegressionClassifier class.
    """
    def __init__(self, input_dim, num_classes):
        super(BlackBoxClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # Returns logits (unnormalized scores)
        return self.linear(x)


class LogisticRegressionClassifier:
    def __init__(self, input_dim, num_classes):
        # Initialize weights and bias
        # Weights: (input_dim, num_classes), Bias: (num_classes)
        self.weights = torch.randn(input_dim, num_classes, requires_grad=False) * 0.01
        self.bias = torch.zeros(num_classes, requires_grad=False)

    def forward(self, x):
        # TODO: implement the logistic regression as z = W^T * x + b. Return z.
        # Hint: this should only require one line of code!
        # STUDENT START ---------------------------------
        # [cite_start]z = W^T * x + b [cite: 237]
        return torch.matmul(x, self.weights) + self.bias
        # STUDENT END -----------------------------------

    def softmax(self, logits):
        # TODO: implement softmax. You may *not* use torch.nn.softmax or any
        # similar function. You may use torch.exp if you wish.
        # STUDENT START --------------------------------
        # [cite_start]Implement softmax manually using torch.exp [cite: 238]
        # Shift logits for numerical stability
        exp_logits = torch.exp(logits - torch.max(logits))
        return exp_logits / torch.sum(exp_logits)
        # STUDENT END ----------------------------------

    def predict(self, x):
        logits = self.forward(x)
        probs = self.softmax(logits)
        return torch.argmax(probs).item()


def train_logistic_regression(train_data, dev_data, featurizer, num_classes=4, lr=0.01, epochs=5,
                              method="bow"):
    input_dim = featurizer.vocab_size
    if method == "lr":
        model = LogisticRegressionClassifier(input_dim, num_classes)
    elif method == "bow":
        model = BlackBoxClassifier(input_dim, num_classes)
    
    print("Training logistic regression...")
    
    for epoch in range(epochs):
        shuffled_train = train_data.copy()
        random.shuffle(shuffled_train)
        total_loss = 0
        
        for ex in shuffled_train:
            x = featurizer.get_feature_vector(ex.text) # (vocab_size,)
            y_true = ex.label
            
            # 1. Call the forward function and compute the probability
            # of each class according to the model.
            logits = model.forward(x)
            probs = model.softmax(logits)
            
            # TODO: 2. Compute the negative log likelihood loss.
            # STUDENT START ----------------------------
            # Loss = -log(probability of true class)
            loss = -torch.log(probs[y_true])
            total_loss += loss.item()
            # STUDENT END ------------------------------
            
            # TODO: 3. Compute the gradient for the weights, and the gradient for
            # for the bias. You may not use .backward().
            # STUDENT START ----------------------------
            # [cite_start]3. Compute gradients [cite: 240]
            # Gradient of Loss w.r.t logits (z) is (probs - y_onehot)
            grad_z = probs.clone()
            grad_z[y_true] -= 1.0

            # grad_W = outer_product(x, grad_z)
            grad_weights = torch.outer(x, grad_z)
            grad_bias = grad_z
            # STUDENT END ------------------------------
            
            # TODO: 4. Update the parameters by multiplying the gradients you
            # derived in the previous step by the learning rate, and then subtracting
            # them from the weights and biases. You will need at least 1 line to update the
            # weight matrix, and at least 1 line to update the bias.
            # STUDENT START ----------------------------
            # [cite_start]4. Update parameters (SGD) [cite: 240]
            model.weights -= lr * grad_weights
            model.bias -= lr * grad_bias
            # STUDENT END ------------------------------
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data):.4f}")
        
    return model


def train_torch_model(train_data, dev_data, featurizer, num_classes=4, lr=0.01, epochs=5):
    """
    Pre-provided gradient descent function using PyTorch's optimizer and loss.
    """
    input_dim = featurizer.vocab_size
    model = BlackBoxClassifier(input_dim, num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    print("Training Built-in PyTorch Model...")
    
    # This is an example of a training loop. Here, we're using only black-box
    # built-in PyTorch functions. You will implement the underlying functionality
    # of these functions as part of Task 2.
    for epoch in range(epochs):
        shuffled_train = train_data.copy()
        random.shuffle(shuffled_train)
        model.train() # Set model to training mode
        total_loss = 0
        
        for ex in shuffled_train:
            x = featurizer.get_feature_vector(ex.text)
            x_tensor = x.unsqueeze(0)
            y_tensor = torch.tensor([ex.label], dtype=torch.long)
            optimizer.zero_grad()
            logits = model(x_tensor)
            loss = criterion(logits, y_tensor)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data):.4f}")
    
    # TODO: You're given the weight matrix of your trained model, which is of
    # shape (C, V), where C is the number of classes (here, 4) and V is
    # the vocabulary size. For each class, you will get the top-5 weight indices,
    # and print out the tokens they correspond to. No need to return anything here;
    # just print out the top weights/tokens and put them in your written report.
    # STUDENT START ----------------------------------
    weights = model.linear.weight

    print("\nTop 5 features per class:")
    # Get the weights from the model's linear layer
    weights = model.linear.weight.detach()
    classes = ["World", "Sports", "Business", "Tech"]

    for i in range(num_classes):
        # Find the indices of the highest 5 weights for this class
        top_indices = torch.topk(weights[i], 5).indices
        top_words = []
        for idx in top_indices:
            # Convert index back to word using the featurizer's inverse vocabulary
            word = featurizer.inverse_vocab.get(idx.item(), "UNK")
            top_words.append(word)
        print(f"{classes[i]}: {top_words}")
    # STUDENT END ------------------------------------
        
    return model
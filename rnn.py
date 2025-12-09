import torch
from torch import nn
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


class RNNCell:
    """Custom implementation of without using torch module API."""
    def __init__(self, fan_in, fan_out):
        self.Wih = torch.randn(fan_in, fan_out)    # Input to hidden weight matrix
        self.Whh = torch.randn(fan_out, fan_out)   # Hiddent to hidden weight matrix
        self.bh = torch.randn(fan_out)             # Bias weight vector    
        self.init()

    def parameters(self):
        return self.Wih, self.Whh, self.bh

    def init(self):
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='tanh')
            else:
                p.data *= 0
            p.requires_grad = True

    def forward(self, x, h):
        return torch.tanh(x @ self.Wih + h @ self.Whh + self.bh) # The output will be the next hidden state
    
    __call__ = forward

class CustomRNN:
    def __init__(self, input_size, hidden_size, num_layers=2):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = [RNNCell(input_size, hidden_size)] + [
            RNNCell(hidden_size, hidden_size)
            for _ in range(num_layers - 1)
        ]

    def parameters(self):
        p = []
        for cell in self.cells:
            p += cell.parameters()
        return p

    def forward(self, x, hx_n):
        B, T, C = x.shape
        outs = []

        if hx_n is None:
            hx_n = torch.zeros(self.num_layers, B, self.hidden_size)

        for t in range(T):
            # List to store all the hidden states
            new_h_n = []

            x_t = x[:, t, :]
            for n, cell in enumerate(self.cells):
                x_t = cell(x_t, hx_n[n])
                new_h_n.append(x_t)

            # Next t will use the t-1 hidden states
            hx_n = new_h_n
            # Keep only the last hidden state of each t
            outs.append(x_t) # (seq_len, batch_size, hidden_size)

        # Return outputs and the last hidden states
        return torch.stack(outs, dim=1), torch.stack(hx_n)
    
    __call__ = forward


class CustomAutoregressiveTextModel:
    character_embed_size = 64
    hidden_size = 256

    def __init__(self, vocab_size):
        self.embed_table = nn.Embedding(
            vocab_size, 
            self.character_embed_size
        )
        self.rnn = CustomRNN(
            self.character_embed_size, 
            self.hidden_size
        )
        self.out = nn.Linear(self.hidden_size, vocab_size)

    def parameters(self):
        return list(self.embed_table.parameters()) + self.rnn.parameters() + list(self.out.parameters())

    def forward(self, idx, hx=None, targets=None):
        # Get mebeddigs
        x = self.embed_table(idx)
        rnn_out, hx = self.rnn(x, hx) # (batch_size, seq_len, channels)
        logits = self.out(rnn_out) # (batch_size, seq_len, vocab_size)

        if targets is not None:
            B, T, C = logits.shape
            # target size is (batch_size, seq_len)
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return (logits, hx), loss
    
    __call__ = forward

    def generate(self, x, max_new_tokens=1000, temperature=1.0):
        hx = None
        with torch.no_grad():
            for t in range(max_new_tokens):
                x = self.embed_table(x)
                rnn_out, hx = self.rnn(x, hx)
                logits = self.out(torch.layer_norm(rnn_out, (self.hidden_size,)))[:, -1, :] # Take last token
                # temp < 1, larger logits which means more confident and distribution sharper
                probs = F.softmax(logits / temperature, dim=-1)
                x = torch.multinomial(probs, num_samples=1)
                yield x

class AutoregressiveTextModel(nn.Module):
    character_embed_size = 64
    hidden_size = 2048

    def __init__(self, vocab_size):
        super().__init__()
        self.embed_table = nn.Embedding(
            vocab_size, 
            self.character_embed_size
        )
        # Use torch RNN
        self.rnn = nn.RNN(
            self.character_embed_size, 
            self.hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.out = nn.Linear(self.hidden_size, vocab_size)

    forward = CustomAutoregressiveTextModel.forward
    generate = CustomAutoregressiveTextModel.generate


def get_model(vocab_size, custom=False):
    if custom:
        return CustomAutoregressiveTextModel(vocab_size)
    else:
        return AutoregressiveTextModel(vocab_size)

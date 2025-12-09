import json
import torch


train_json = json.loads(open("train.json").read())
val_json = json.loads(open("validation.json").read())


categories = set()
characters = set()
for book in train_json + val_json:
    categories.add(book["category"])
    characters = characters.union(set(book["EN"]))

cat_token = lambda cat: "<" + cat.replace(" ", "_").lower() + ">"
categories = [cat_token(cat) for cat in categories]
# special tokens for category plus single  characters
tokens = categories + list(sorted(characters))

vocab_size = len(tokens)

print(f"Vocab size: {vocab_size}")
print(f"Tokens: {tokens}")

stoi = {ch: i for i, ch in enumerate(tokens)}
itos = {i: ch for i, ch in enumerate(tokens)}
encode = lambda s, cat=None: ([stoi[cat]] if cat else []) + [stoi[c] for c in s]
decode = lambda idxs: ''.join([itos[i] for i in idxs])

print("Test encode-decode")
print(encode("Hello!", "<adventure>"))
print(decode(encode("Hello!", "<adventure>")))

train_data = [
    (
        torch.tensor(encode("", cat_token(book["category"]))), 
        torch.tensor(encode(book["EN"]))
    ) 
    for book in val_json
]
val_data = [
    (
        torch.tensor(encode("", cat_token(book["category"]))), 
        torch.tensor(encode(book["EN"]))
    ) 
    for book in val_json
]

def get_batch(split, batch_size, block_size):
    # The category token will be added
    block_size -= 1

    x, y = [], []
    data = train_data if split == "train" else val_data
    booki = torch.randint(0, len(data) - 1, (batch_size,))
    for i in booki:
        cat, seq = data[i]
        seq_start = torch.randint(0, len(seq) - block_size, (1,)).item()
        seq_x = seq[seq_start:seq_start+block_size].clone()
        seq_y = seq[seq_start:seq_start+block_size+1].clone()
        # Concat category to sequence
        x.append(torch.cat([cat, seq_x]))
        y.append(seq_y)
    return torch.stack(x), torch.stack(y)

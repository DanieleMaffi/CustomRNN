import rnn
import dataset
import torch


BATCH_SIZE = 32
BLOCK_SIZE = 1024
TRAIN = False
LOAD = True

model = rnn.get_model(dataset.vocab_size, custom=False)
optim = torch.optim.Adam(model.parameters(), lr=1e-5)


def save(step, name):
    torch.save({
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
        "step": step
    }, name)


def load(name):
    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint["model_state"])
    optim.load_state_dict(checkpoint["optim_state"])
    return checkpoint["step"]

@torch.no_grad()
def validate_split(split, steps=1000):
    lossi = []
    h = None
    for step in range(steps):
        xb, yb = dataset.get_batch(split, BATCH_SIZE, BLOCK_SIZE)
        _, loss = model(xb, h, yb)
        lossi.append(loss.item())
    return torch.tensor(lossi).mean().item()

def train(steps, max_norm=2.0):
    h = None
    for step in range(steps):
        xb, yb = dataset.get_batch("train", BATCH_SIZE, BLOCK_SIZE)

        if h is not None:
            h = h.detach()

        (logits, h), loss = model(xb, h, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optim.step()

        if step % 100 == 0:
            for p in model.parameters():
                print(f"{p.grad.data.mean().item():.5f}, {p.grad.data.std().item():.5f}")
            print(f"step {step}, loss {loss.item()}")


def generate(category, max_new_tokens=1000, temperature=1.0):
    start_token = torch.tensor(dataset.encode("", category)).unsqueeze(0)
    for out in model.generate(start_token, max_new_tokens=max_new_tokens, temperature=temperature):
        print(dataset.decode([out.item()]), end='', flush=True)


steps = 5000
if LOAD:
    load("checkpoint.pth")
    print(f"Loading checkpoint at step {steps}")
if TRAIN:
    train(steps)
    save(steps, "checkpoint.pth")

# print(validate_split("train", steps))
print(validate_split("val"))
# Kick off generation with just the category
generate("<science_fiction>", max_new_tokens=1000000, temperature=0.5)

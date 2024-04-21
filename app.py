from helpers.config import LLMConfig, TrainingConfig, get_device
from helpers.trainer import train
from helpers.dataset import NextTokenPredictionDataset
from model.tokenizer import Tokenizer, train_tokenizer
from model.llm import LLM
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append("..")


llm_config = LLMConfig(
    vocab_size=2_000,
    context_size=128,
    dim_emb=256,
    num_layers=4,
    num_heads=8,
    emb_dropout=0.0,
    ffd_dim_hidden=4 * 256,
    ffd_bias=False,
)

train_config = TrainingConfig(
    retrain_tokenizer=False,
    device=get_device(),
    batch_size=64,
    learning_rate=3e-4,
    weight_decay=1e-5,
    max_steps=4_000,
    log_frequency=1,
)


input_file = "data/tinyshakespeare.txt"


output_file = Path(input_file).with_suffix(".model")

if not output_file.exists() or train_config.retrain_tokenizer:
    train_tokenizer(input_file, llm_config.vocab_size)

tokenizer = Tokenizer(str(output_file))


sentence = (
    "The role of the tokenizer is to build a mapping between a sentences represented as a string and token indices."
)
print(tokenizer.sp.EncodeAsPieces(sentence))

assert tokenizer.decode(tokenizer.encode(sentence)) == sentence

# This helper class allow to generate batches of inputs and targets where targets last element is the next token to predict
ds_train = NextTokenPredictionDataset(
    input_file, llm_config.context_size, tokenizer)

X, y = ds_train.get_batch(batch_size=1)

print(X.shape, y.shape)

model = LLM(
    vocab_size=tokenizer.vocab_size,
    context_size=llm_config.context_size,
    dim_emb=llm_config.dim_emb,
    num_layers=llm_config.num_layers,
    attn_num_heads=llm_config.num_heads,
    emb_dropout=llm_config.emb_dropout,
    ffd_hidden_dim=llm_config.ffd_dim_hidden,
    ffd_bias=llm_config.ffd_bias,
)

params_size = sum(p.nelement() * p.element_size() for p in model.parameters())
buffer_size = sum(p.nelement() * p.element_size() for p in model.buffers())
size = (params_size + buffer_size) / 1024**2

print(f"total params: {sum(p.numel() for p in model.parameters()):,d}")
print(f"model size: {size:.3f}MB")

# print(model)

loss_history = train(
    model,
    ds_train,
    train_config.device,
    batch_size=train_config.batch_size,
    lr=train_config.learning_rate,
    max_steps=train_config.max_steps,
    weight_decay=train_config.weight_decay,
    log_every=train_config.log_frequency,
)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(len(loss_history["train_loss"])), loss_history["train_loss"])
ax.set_xlabel("step")
ax.set_ylabel("cross entropy loss")
ax.grid(axis="y")

plt.savefig('training-graphs/training-graph.png')


# empty prompt to generate random stuff
prompt = torch.full((1, llm_config.context_size),
                    tokenizer.pad_id, dtype=torch.int32)
prompt[..., 0] = tokenizer.bos_id
prompt = prompt.to(train_config.device)

out = model.generate(prompt, max_seq_len=128)
print(tokenizer.decode(out))

# generate from a prompt
prompt = (
    tokenizer.encode(
        "what is humour?",
        beg_of_string=True,
        pad_seq=True,
        seq_len=llm_config.context_size,
    )
    .view(1, -1)
    .to(train_config.device)
)
out = model.generate(prompt, max_seq_len=128)

print(tokenizer.decode(out))

print("Done")

# Define a function to interact with the chatbot


def interact_with_chatbot(model, tokenizer, prompt_prefix, max_seq_len=128):
    while True:
        # Prompt user for input
        user_input = input("You: ")

        # Encode user input and add it to the prompt
        prompt = tokenizer.encode(
            prompt_prefix + user_input,
            beg_of_string=True,
            pad_seq=True,
            seq_len=llm_config.context_size,
        ).view(1, -1).to(train_config.device)

        # Generate response from the model
        response = model.generate(prompt, max_seq_len=max_seq_len)

        # Decode the generated response and print it
        print("Chatbot:", tokenizer.decode(response))


# Interaction with the chatbot
interact_with_chatbot(model, tokenizer, "User: ")

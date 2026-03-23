import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import time

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_NAME = "bentrevett/multi30k"
DATASET_SPLIT = "train"
SOURCE_KEY = "en"
TARGET_KEY = "de"
TOKENIZER_NAME = "bert-base-multilingual-cased"

SUBSET_SIZE = 1000
MAX_SRC_LEN = 24
MAX_TGT_LEN = 24
BATCH_SIZE = 16

D_MODEL = 128
NUM_HEADS = 4
D_FF = 256
NUM_LAYERS = 2

EPOCHS = 12
LEARNING_RATE = 1e-3


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attention_output = scaled_dot_product_attention(Q, K, V, mask)
        output = self.combine_heads(attention_output)
        return self.W_o(output)


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, src_mask=None):
        attention_output = self.self_attn(x, x, x, src_mask)
        x = self.add_norm1(x, attention_output)

        ffn_output = self.ffn(x)
        x = self.add_norm2(x, ffn_output)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm2 = AddNorm(d_model)

        self.ffn = PositionWiseFFN(d_model, d_ff)
        self.add_norm3 = AddNorm(d_model)

    def forward(self, y, memory, tgt_mask=None, src_mask=None):
        masked_attention_output = self.masked_self_attn(y, y, y, tgt_mask)
        y = self.add_norm1(y, masked_attention_output)

        cross_attention_output = self.cross_attn(y, memory, memory, src_mask)
        y = self.add_norm2(y, cross_attention_output)

        ffn_output = self.ffn(y)
        y = self.add_norm3(y, ffn_output)
        return y


class TransformerFromScratch(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=128,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_len=512
    ):
        super().__init__()

        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask=None):
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(self, tgt, memory, tgt_mask=None, src_mask=None):
        y = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        y = self.positional_encoding(y)

        for layer in self.decoder_layers:
            y = layer(y, memory, tgt_mask, src_mask)

        return y

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        y = self.decode(tgt, memory, tgt_mask, src_mask)
        logits = self.output_layer(y)
        return logits


def create_padding_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(size, device):
    return torch.tril(
        torch.ones(size, size, dtype=torch.bool, device=device)
    ).unsqueeze(0).unsqueeze(0)


def create_decoder_mask(tgt, pad_idx):
    pad_mask = create_padding_mask(tgt, pad_idx)
    causal_mask = create_causal_mask(tgt.size(1), tgt.device)
    return pad_mask & causal_mask


class ParallelTextDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def load_parallel_pairs():
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    limit = min(SUBSET_SIZE, len(dataset))
    subset = dataset.select(range(limit))

    pairs = []
    for item in subset:
        src_text = item[SOURCE_KEY].strip()
        tgt_text = item[TARGET_KEY].strip()

        if src_text and tgt_text:
            pairs.append((src_text, tgt_text))

    return pairs


def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    tokens_to_add = []
    if "<START>" not in tokenizer.get_vocab():
        tokens_to_add.append("<START>")
    if "<EOS>" not in tokenizer.get_vocab():
        tokens_to_add.append("<EOS>")

    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    return tokenizer


def make_collate_fn(tokenizer, start_id, eos_id, pad_id):
    def collate_fn(batch):
        src_batch = []
        tgt_input_batch = []
        tgt_output_batch = []

        for src_text, tgt_text in batch:
            src_ids = tokenizer.encode(
                src_text,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_SRC_LEN
            )

            tgt_ids = tokenizer.encode(
                tgt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_TGT_LEN - 2
            )

            tgt_input_ids = [start_id] + tgt_ids
            tgt_output_ids = tgt_ids + [eos_id]

            src_batch.append(torch.tensor(src_ids, dtype=torch.long))
            tgt_input_batch.append(torch.tensor(tgt_input_ids, dtype=torch.long))
            tgt_output_batch.append(torch.tensor(tgt_output_ids, dtype=torch.long))

        src_tensor = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
        tgt_input_tensor = pad_sequence(tgt_input_batch, batch_first=True, padding_value=pad_id)
        tgt_output_tensor = pad_sequence(tgt_output_batch, batch_first=True, padding_value=pad_id)

        return src_tensor, tgt_input_tensor, tgt_output_tensor

    return collate_fn


def grad_norm(model):
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            norm = param.grad.detach().data.norm(2)
            total += norm.item() ** 2
    return total ** 0.5


def greedy_decode(model, tokenizer, src_text, pad_id, start_id, eos_id):
    model.eval()

    with torch.no_grad():
        src_ids = tokenizer.encode(
            src_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_SRC_LEN
        )

        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
        src_mask = create_padding_mask(src_tensor, pad_id)
        memory = model.encode(src_tensor, src_mask)

        generated = [start_id]

        for _ in range(MAX_TGT_LEN):
            tgt_tensor = torch.tensor([generated], dtype=torch.long, device=DEVICE)
            tgt_mask = create_decoder_mask(tgt_tensor, pad_id)

            decoder_output = model.decode(tgt_tensor, memory, tgt_mask, src_mask)
            logits = model.output_layer(decoder_output)

            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()

            if next_token_id == eos_id:
                break

            generated.append(next_token_id)

    return tokenizer.decode(generated[1:], skip_special_tokens=True).strip()


def main():
    start_time = time.time()
    print(f"Dispositivo: {DEVICE}")
    print("Carregando dataset...")

    train_pairs = load_parallel_pairs()
    print(f"Quantidade de pares usados no treino: {len(train_pairs)}")

    print("Carregando tokenizer...")
    tokenizer = build_tokenizer()

    start_id = tokenizer.convert_tokens_to_ids("<START>")
    eos_id = tokenizer.convert_tokens_to_ids("<EOS>")
    pad_id = tokenizer.pad_token_id

    dataset = ParallelTextDataset(train_pairs)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, start_id, eos_id, pad_id)
    )

    model = TransformerFromScratch(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=max(MAX_SRC_LEN, MAX_TGT_LEN) + 5
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Iniciando treinamento...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_grad = 0.0

        for src_batch, tgt_input_batch, tgt_output_batch in dataloader:
            src_batch = src_batch.to(DEVICE)
            tgt_input_batch = tgt_input_batch.to(DEVICE)
            tgt_output_batch = tgt_output_batch.to(DEVICE)

            src_mask = create_padding_mask(src_batch, pad_id)
            tgt_mask = create_decoder_mask(tgt_input_batch, pad_id)

            logits = model(src_batch, tgt_input_batch, src_mask, tgt_mask)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output_batch.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            batch_grad = grad_norm(model)
            optimizer.step()

            total_loss += loss.item()
            total_grad += batch_grad

        avg_loss = total_loss / len(dataloader)
        avg_grad = total_grad / len(dataloader)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | GradNorm: {avg_grad:.4f}")

    sample_src, sample_tgt = train_pairs[0]
    generated_translation = greedy_decode(
        model,
        tokenizer,
        sample_src,
        pad_id,
        start_id,
        eos_id
    )

    print("\n=== OVERFITTING TEST ===")
    print("Frase de entrada:", sample_src)
    print("Tradução real:", sample_tgt)
    print("Tradução gerada:", generated_translation)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTempo total de execução: {total_time:.2f} segundos")

if __name__ == "__main__":
    main()
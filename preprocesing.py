

"""
Los pipelines se utilizan para predecir texto nuevo
"""

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
import torch

# tokenizador básico en inglés
tokenizer = get_tokenizer("basic_english")

# Vocabulario
vocab = None

# Función que genera un vocabulario a partir de un iterador
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text.lower())

# Genera un vocabulario 
def build_vocab():
    global vocab
    print("INFO Construyendo vocabulario...")
    train_iter = AG_NEWS(split="train")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    print(f"[DEBUG] Vocab size: {len(vocab)}")
    print(f"[DEBUG] Sample tokens: {list(vocab.get_stoi().keys())[:10]}")
    return vocab

# Convierte texto en tokens y después en índices
def text_pipeline(x):
    return vocab(tokenizer(x))

# Convierte las etiquetas a enteros, AG_NEWS tiene 4 clases
def label_pipeline(x):
    return int(x) - 1


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets

# Exponemos la variable global vocab

def get_vocab():
    if vocab is None:
        raise ValueError("Vocabulario aún no ha sido construido. Llama primero a build_vocab().")
    return vocab


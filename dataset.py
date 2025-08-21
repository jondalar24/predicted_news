

from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.dataset import random_split
from torchtext.datasets import AG_NEWS

def load_datasets(split_ratio=0.95):
    # Obtener iteradores
    train_iter, test_iter = AG_NEWS()

    # Convertir a map-style (permite indexado y uso de len)
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)

    # Cálculo de tamaño del split
    num_train = int(len(train_dataset) * split_ratio)

    # Split en entrenamiento y validación
    split_train_, split_valid_ = random_split(
        train_dataset,
        [num_train, len(train_dataset) - num_train]
    )

    return split_train_, split_valid_, test_dataset

def load_datasets_for_vocab():
    iter = AG_NEWS(split="train")
    return iter


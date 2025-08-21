

from dataset import load_datasets
from preprocesing import build_vocab, collate_batch, get_vocab
from model import TextClassificationModel
from config import *
from train import train_model
from evaluate import evaluate
from utils import plot
import torch
from predict import predict_text
from torch.utils.data import DataLoader



#EmbeddingBag no está preparada para MPS. Descomenta esta linea el día que si lo esté
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")


def main():

    # Iniciamos el proceso de carga generando el vocabulario en preprocessing
    build_vocab()
    
    # Creamos el vocabulario    
    vocab = get_vocab()

    # Carga dataset
    split_train_, split_valid_, text_dataset = load_datasets()    

    # debug
    if vocab:
        print("El vocabulario existe y tiene una longitud de:")
        print(len(vocab))
    
    # Cargadores de batches e implementación de collate_fn
    train_loader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(text_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # Definir modelo
    model = TextClassificationModel(len(vocab), EMBEDDING_SIZE, NUM_CLASS).to(device)

    # Entrenar:
    losses, accs = train_model(model, train_loader, valid_loader, device)

    # Evaluar:    
    test_accuracy = evaluate(test_loader, model, device)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Visualizar
    plot(losses, accs)

    while True:
        print("\nIntroduce una noticia para clasificar (o escribe 'exit' para salir):")
        new_text = input("> ")
        if new_text.lower() == 'exit':
            break
        category = predict_text(model, new_text, device)
        print(f"\nCategoría predicha: {category}")


if __name__ == "__main__":
    main()
    


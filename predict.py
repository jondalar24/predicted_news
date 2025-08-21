
import torch
from preprocesing import text_pipeline, get_vocab

# Mapear las clases predichas al nombre de categoría
ag_news_label = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

def predict_text(model, text, device):
    model.eval()
    vocab = get_vocab()  # Recuperamos el vocabulario global
    with torch.no_grad():
        # Preprocesado del texto
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64).to(device)
        offset = torch.tensor([0]).to(device)
        # Predicción
        output = model(processed_text, offset)
        pred_class = output.argmax(1).item()
        return ag_news_label[pred_class]

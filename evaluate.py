"""
Evalúa el modelo sobre cualquier dataloader sin necesitar grad
Usa argmax para comparar la predicción contra la eitqueta
Devuelve la accuracy
"""
#Librerías
import torch



def evaluate(dataloader,model, device):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for label, text, offsets in dataloader:
            label, text, offsets = label.to(device), text.to(device), offsets.to(device)
            predicted_label = model(text, offsets)
            acc = (predicted_label.argmax(1) == label).sum().item()
            total_acc += acc
            total_count += label.size(0)

            # Debug por batch
            #print(f"[DEBUG] Batch Acc: {acc}/{label.size(0)}")
    return total_acc / total_count



#  Clasificador de Noticias con Deep Learning (PyTorch + TorchText)

Este proyecto aplica técnicas de procesamiento de lenguaje natural (NLP) con deep learning para clasificar noticias en inglés utilizando el dataset AG_NEWS. El modelo ha sido implementado con PyTorch, haciendo uso de `nn.EmbeddingBag` para gestionar entradas de longitud variable de forma eficiente.

---

##  Teoría Aplicada

El objetivo de este sistema es **clasificar artículos de prensa en 4 categorías**: `World`, `Sports`, `Business` y `Sci/Tech`. Para ello, se realiza un pipeline de procesamiento que convierte texto en tokens mediante un vocabulario indexado. Posteriormente, las palabras son convertidas a vectores semánticos (embeddings), permitiendo al modelo detectar patrones y relaciones en el lenguaje.

La arquitectura se basa en:
- **Tokenización básica** (`basic_english`).
- **Vocabulario con indexación**.
- **Embeddings con `nn.EmbeddingBag`**, que permite procesar lotes de textos de distinta longitud de manera eficiente.
- **Modelo lineal con `nn.Linear`** para clasificación.
- **Entrenamiento supervisado** con `CrossEntropyLoss` y `SGD`.

---

##  Requisitos

Este proyecto ha sido desarrollado y probado en **Python 3.10**, ya que versiones superiores pueden presentar incompatibilidades con ciertos módulos de PyTorch. Para gestionar versiones de Python, se recomienda usar **`pyenv`**:

```bash
pyenv install 3.10.13
pyenv local 3.10.13
```

---

##  Instalación de dependencias

Asegúrate de tener un entorno virtual activo (por ejemplo con `venv` o `virtualenv`) y luego ejecuta:

```bash
pip install -r requirements.txt
```

Esto instalará `torch`, `torchtext`, `tqdm`, `matplotlib`, `scikit-learn`, `plotly` y otros módulos necesarios para entrenamiento y visualización.

---

##  Cómo probar el modelo

Una vez entrenado, puedes probar el modelo con nuevos textos utilizando la función `predict()` que convierte texto crudo en una predicción de categoría:

```python
article = "The new tech breakthrough by Apple could change the future of AI."
result = predict(article, text_pipeline)
print("Predicted category:", result)
```

También puedes visualizar cómo el modelo entiende los textos con t-SNE en 3D (requiere `plotly`).

---

##  Estructura del Proyecto

```
news_classifier/
│
├── main.py                # Punto de entrada principal
├── dataset.py             # Carga y división del dataset
├── preprocessing.py       # Tokenización, vocabulario y batch processing
├── model.py               # Definición del modelo con EmbeddingBag
├── train.py               # Lógica de entrenamiento
├── evaluate.py            # Evaluación de precisión
├── utils.py               # Visualización de métricas
├── requirements.txt       # Dependencias
└── README.md              # Este archivo
```

---

## Resultados

El modelo alcanza una precisión de **~81% en 10 épocas**, utilizando `CrossEntropyLoss` y optimización con `SGD`. Se ha desactivado el scheduler de learning rate para asegurar resultados consistentes en primeras pruebas.

---

##  Contribuye

¿Quieres mejorar la clasificación? ¿Adaptarlo a otro idioma o categoría? ¡Modifica el código y comparte tus resultados! Cualquier sugerencia, PR o mejora es bienvenida. ¡La comunidad aprende en colaboración!

---

© 2025 - Proyecto educativo personal para explorar Deep Learning en NLP con PyTorch.

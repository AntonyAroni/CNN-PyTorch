import os
import pandas as pd
from PIL import Image
import numpy as np

# Mapeo de nombres de clase a etiquetas numéricas
class_to_label = {
    'basophil': 0,
    'eosinophil': 1,
    'erythroblast': 2,
    'ig': 3,
    'lymphocyte': 4,
    'monocyte': 5,
    'neutrophil': 6,
    'platelet': 7
}

data = []
for label in os.listdir("bloodmnist_images"):
    path = os.path.join("bloodmnist_images", label)
    for img_file in os.listdir(path):
        img = Image.open(os.path.join(path, img_file)).resize((28, 28))
        img_array = np.array(img).astype(np.float32) / 255.0  # Normaliza a [0, 1]
        row = [class_to_label[label.lower()]] + img_array.flatten().tolist()  # Usa el mapeo
        data.append(row)

# Guarda el CSV
columns = ["label"] + [f"pixel_{i}" for i in range(28*28*3)]
df = pd.DataFrame(data, columns=columns)
df.to_csv("blood_dataset.csv", index=False)
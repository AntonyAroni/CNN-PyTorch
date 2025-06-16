import numpy as np
from PIL import Image
import os
from medmnist import BloodMNIST

# Configuración
download = True  # Descargar si no existe
output_dir = "bloodmnist_images"  # Carpeta de salida

# Nombres de las clases (según BloodMNIST)
class_names = [
    'basophil',
    'eosinophil',
    'erythroblast',
    'ig',
    'lymphocyte',
    'monocyte',
    'neutrophil',
    'platelet'
]

# Crear directorio de salida
os.makedirs(output_dir, exist_ok=True)

# 1. Cargar el dataset desde medmnist
print("Cargando BloodMNIST...")
bloodmnist = BloodMNIST(split="train", download=download)

# 2. Extraer imágenes y etiquetas
images = bloodmnist.imgs  # Array de numpy (N, 28, 28, 3)
labels = bloodmnist.labels.flatten()  # Array de etiquetas (N,)

print(f"Total de imágenes: {len(images)}")
print(f"Total de etiquetas: {len(labels)}")

# 3. Guardar imágenes en subcarpetas por clase
print(f"\nGuardando imágenes en '{output_dir}'...")
for i, (image, label) in enumerate(zip(images, labels)):
    # Crear subcarpeta por clase
    class_dir = os.path.join(output_dir, class_names[label])
    os.makedirs(class_dir, exist_ok=True)

    # Convertir y guardar como PNG
    img = Image.fromarray(image)
    img.save(os.path.join(class_dir, f"img_{i:05d}.png"))

    # Mostrar progreso cada 100 imágenes
    if (i + 1) % 100 == 0:
        print(f"Procesadas: {i + 1}/{len(images)} imágenes")

# 4. Resumen final
print("\n" + "=" * 50)
print(f"¡Conversión completada!")
print(f"Imágenes guardadas en: {os.path.abspath(output_dir)}")
print("Estructura de carpetas creada:")
for class_name in class_names:
    class_path = os.path.join(output_dir, class_name)
    if os.path.exists(class_path):
        print(f"  {class_name}: {len(os.listdir(class_path))} imágenes")
print("=" * 50)
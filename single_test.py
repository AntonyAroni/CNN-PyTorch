import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# 1. Copia exacta de tu arquitectura (de model.py)
class BloodMNISTCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# 2. Función para cargar el modelo
def load_model(model_path, device='cpu'):
    model = BloodMNISTCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# 3. Preprocesamiento (igual que en tu entrenamiento)
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


# 4. Predicción
def predict_image(model, image_path, class_names, device='cpu'):
    image = Image.open(image_path).convert('RGB')
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities).item()

    return {
        'class': predicted_class,
        'class_name': class_names[predicted_class],
        'probabilities': probabilities.squeeze().tolist()
    }


# 5. Visualización
def plot_prediction(image_path, prediction, class_names):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f'Predicción: {prediction["class_name"]}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    bars = plt.barh(class_names, prediction['probabilities'])
    plt.xlim(0, 1)
    plt.title('Probabilidades por clase')
    plt.xlabel('Probabilidad')

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.05, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', ha='left', va='center')

    plt.tight_layout()
    plt.show()


# --- Configuración principal ---
if __name__ == "__main__":
    # Ajusta estas rutas
    MODEL_PATH = "checkpoints/best_model_20250615_162551.pth"  # Ruta a tu modelo .pth
    IMAGE_PATH = "test/ig/img_00046.png"  # Ruta a una imagen de prueba (28x28 RGB)
    CLASS_NAMES = ['basophil', 'eosinophil', 'erythroblast', 'ig',
                   'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Cargar modelo
    print("Cargando modelo...")
    model = load_model(MODEL_PATH, device)

    # Predecir
    print(f"\nEvaluando imagen: {IMAGE_PATH}")
    prediction = predict_image(model, IMAGE_PATH, CLASS_NAMES, device)

    # Resultados
    print(f"\nResultado:")
    print(f"- Clase predicha: {prediction['class_name']}")
    print("- Probabilidades:")
    for name, prob in zip(CLASS_NAMES, prediction['probabilities']):
        print(f"  {name}: {prob:.4f}")

    # Visualización
    plot_prediction(IMAGE_PATH, prediction, CLASS_NAMES)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import zipfile
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Configuración de entrenamiento
TRAIN_MODEL = True  # Cambiar a False si solo quieres evaluar un modelo existente
MODEL_PATH = 'simpsons_mnist_cnn_model.pth'
CONTINUE_TRAINING = False  # Cambiar a True para continuar entrenamiento desde checkpoint
USE_RGB = True  # True para RGB, False para escala de grises

# Clase para el dataset personalizado de SimpsonsMNIST
class SimpsonsMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Mapeo de nombres de carpetas a índices
        self.class_to_idx = {
            "bart_simpson": 0,
            "charles_montgomery_burns": 1,
            "homer_simpson": 2,
            "krusty_the_clown": 3,
            "lisa_simpson": 4,
            "marge_simpson": 5,
            "milhouse_van_houten": 6,
            "moe_szyslak": 7,
            "ned_flanders": 8,
            "principal_skinner": 9
        }

        # Cargar imágenes y etiquetas
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        if not USE_RGB:
            image = image.convert('L')  # Convertir a escala de grises
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Función para descargar y extraer el dataset
def download_and_extract_dataset():
    """Extrae el dataset SimpsonsMNIST desde archivos locales"""
    import zipfile
    import os

    # Determinar tipo de dataset
    if USE_RGB:
        dataset_type = "rgb"
        train_zip_file = "rgb-train.zip"
        test_zip_file = "rgb-test.zip"
    else:
        dataset_type = "grayscale"
        train_zip_file = "grayscale-train.zip"
        test_zip_file = "grayscale-test.zip"

    # Crear directorios
    os.makedirs("simpsons_dataset", exist_ok=True)
    train_dir = f"simpsons_dataset/{dataset_type}_train"
    test_dir = f"simpsons_dataset/{dataset_type}_test"

    # Verificar que los archivos ZIP existen
    if not os.path.exists(train_zip_file):
        raise FileNotFoundError(f"No se encontró el archivo {train_zip_file} en el directorio actual")

    if not os.path.exists(test_zip_file):
        raise FileNotFoundError(f"No se encontró el archivo {test_zip_file} en el directorio actual")

    # Extraer conjunto de entrenamiento
    if not os.path.exists(train_dir):
        print(f"Extrayendo conjunto de entrenamiento desde {train_zip_file}...")
        try:
            with zipfile.ZipFile(train_zip_file, 'r') as zip_ref:
                zip_ref.extractall(train_dir)
            print("Conjunto de entrenamiento extraído correctamente!")
        except zipfile.BadZipFile:
            raise Exception(f"El archivo {train_zip_file} está corrupto o no es un ZIP válido")
    else:
        print("Conjunto de entrenamiento ya existe, omitiendo extracción.")

    # Extraer conjunto de prueba
    if not os.path.exists(test_dir):
        print(f"Extrayendo conjunto de prueba desde {test_zip_file}...")
        try:
            with zipfile.ZipFile(test_zip_file, 'r') as zip_ref:
                zip_ref.extractall(test_dir)
            print("Conjunto de prueba extraído correctamente!")
        except zipfile.BadZipFile:
            raise Exception(f"El archivo {test_zip_file} está corrupto o no es un ZIP válido")
    else:
        print("Conjunto de prueba ya existe, omitiendo extracción.")

    # Ajustar las rutas para incluir el subdirectorio 'train'
    train_dir = os.path.join(train_dir, "train")
    test_dir = os.path.join(test_dir, "test")  # Asumiendo que la estructura es similar para test

    return train_dir, test_dir

# Descargar y preparar el dataset
print("Preparando dataset SimpsonsMNIST...")
train_dir, test_dir = download_and_extract_dataset()

# Transformaciones para el dataset
if USE_RGB:
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización RGB
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalización escala de grises
    ])

# Crear datasets
print("Creando datasets...")
train_dataset = SimpsonsMNISTDataset(train_dir, transform=transform)
test_dataset = SimpsonsMNISTDataset(test_dir, transform=transform)

# Crear DataLoaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Tamaño del conjunto de entrenamiento: {len(train_dataset)}")
print(f"Tamaño del conjunto de prueba: {len(test_dataset)}")
print(f"Número de clases: 10")
print(f"Tipo de imagen: {'RGB' if USE_RGB else 'Escala de grises'}")

# Definir la arquitectura CNN
class SimpsonsCNN(nn.Module):
    def __init__(self, num_classes=10, use_rgb=True):
        super(SimpsonsCNN, self).__init__()

        input_channels = 3 if use_rgb else 1

        # Primera capa convolucional
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # Segunda capa convolucional
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # Tercera capa convolucional
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        # Cuarta capa convolucional
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        # Dropout para regularización
        self.dropout = nn.Dropout(0.5)

        # Capas fully connected
        # Para imágenes de 28x28, después de 4 poolings de 2x2: 28/16 = 1.75 -> 1x1
        self.fc1 = nn.Linear(256 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x

# Inicializar el modelo
model = SimpsonsCNN(num_classes=10, use_rgb=USE_RGB).to(device)
print(f"Modelo creado con {sum(p.numel() for p in model.parameters())} parámetros")

# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Función para evaluar el modelo
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels, total_loss / len(data_loader)

# Función de entrenamiento
def train_model(model, train_loader, num_epochs=10):
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        scheduler.step()

        print(f'Época [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print('-' * 50)

    return train_losses, train_accuracies

# Verificar si existe un modelo guardado
model_exists = os.path.exists(MODEL_PATH)

if model_exists and not TRAIN_MODEL:
    print(f"Cargando modelo existente desde {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Modelo cargado exitosamente. Saltando entrenamiento...")
elif model_exists and CONTINUE_TRAINING:
    print(f"Cargando modelo existente para continuar entrenamiento...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Continuando entrenamiento desde checkpoint...")
    train_losses, train_accs = train_model(model, train_loader, num_epochs=10)
elif TRAIN_MODEL:
    print("Entrenando modelo desde cero...")
    train_losses, train_accs = train_model(model, train_loader, num_epochs=10)
    # Guardar el modelo
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModelo guardado como '{MODEL_PATH}'")
else:
    print("No se encontró modelo guardado y TRAIN_MODEL está en False.")
    print("Cambia TRAIN_MODEL = True para entrenar un nuevo modelo.")

# Evaluación final en el conjunto de prueba
print("\nEvaluando en el conjunto de prueba...")
test_preds, test_labels, test_loss = evaluate_model(model, test_loader, device)

# Calcular métricas
accuracy = accuracy_score(test_labels, test_preds)
precision = precision_score(test_labels, test_preds, average='weighted')
recall = recall_score(test_labels, test_preds, average='weighted')
f1 = f1_score(test_labels, test_preds, average='weighted')

# Mostrar resultados finales
print("\n" + "="*60)
print("RESULTADOS FINALES - SIMPSONS MNIST CNN")
print("="*60)
print(f"Tipo de imagen: {'RGB' if USE_RGB else 'Escala de grises'}")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("="*60)

# Nombres de las clases
class_names = [
    'Bart Simpson',
    'Charles Montgomery Burns',
    'Homer Simpson',
    'Krusty the Clown',
    'Lisa Simpson',
    'Marge Simpson',
    'Milhouse Van Houten',
    'Moe Szyslak',
    'Ned Flanders',
    'Principal Skinner'
]

# Reporte detallado de clasificación
print("\nReporte detallado de clasificación:")
print(classification_report(test_labels, test_preds, target_names=class_names))

# Función para plotear resultados
def plot_training_history(train_losses, train_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot de pérdidas
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.set_title('Pérdida durante el entrenamiento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.legend()
    ax1.grid(True)

    # Plot de precisión
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.set_title('Precisión durante el entrenamiento')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Precisión (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Función para plotear matriz de confusión
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión - SimpsonsMNIST')
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Mostrar gráficos solo si se entrenó el modelo
if TRAIN_MODEL or CONTINUE_TRAINING:
    plot_training_history(train_losses, train_accs)

plot_confusion_matrix(test_labels, test_preds, class_names)

# Métricas por clase individual
print("\nMétricas por clase individual:")
precision_per_class = precision_score(test_labels, test_preds, average=None)
recall_per_class = recall_score(test_labels, test_preds, average=None)
f1_per_class = f1_score(test_labels, test_preds, average=None)

for i, class_name in enumerate(class_names):
    print(f"{class_name}:")
    print(f"  Precision: {precision_per_class[i]:.4f}")
    print(f"  Recall:    {recall_per_class[i]:.4f}")
    print(f"  F1 Score:  {f1_per_class[i]:.4f}")
    print("-" * 40)

# Función para mostrar algunas predicciones
def show_sample_predictions(model, test_loader, device, num_samples=8):
    model.eval()

    # Obtener una muestra de datos
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    # Hacer predicciones
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Mostrar las imágenes con sus predicciones
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()

    for i in range(min(num_samples, len(images))):
        img = images[i].cpu()

        # Desnormalizar la imagen para visualización
        if USE_RGB:
            # Desnormalizar RGB
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img.permute(1, 2, 0).numpy()
            img = std * img + mean
            img = np.clip(img, 0, 1)
        else:
            # Desnormalizar escala de grises
            img = img.squeeze().numpy()
            img = (img * 0.5) + 0.5
            img = np.clip(img, 0, 1)

        axes[i].imshow(img, cmap='gray' if not USE_RGB else None)
        true_label = class_names[labels[i].item()]
        pred_label = class_names[predicted[i].item()]
        color = 'green' if labels[i] == predicted[i] else 'red'
        axes[i].set_title(f'Real: {true_label}\nPredicción: {pred_label}',
                          color=color, fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Mostrar algunas predicciones de ejemplo
print("\nEjemplos de predicciones:")
show_sample_predictions(model, test_loader, device)
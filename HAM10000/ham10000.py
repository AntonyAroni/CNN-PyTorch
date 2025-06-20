import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from medmnist import DermaMNIST
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Transformaciones para el dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalización para imágenes en escala de grises
])

# Cargar el dataset DermaMNIST
print("Cargando dataset DermaMNIST...")
train_dataset = DermaMNIST(split='train', transform=transform, download=True)
val_dataset = DermaMNIST(split='val', transform=transform, download=True)
test_dataset = DermaMNIST(split='test', transform=transform, download=True)

# Crear DataLoaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Tamaño del conjunto de entrenamiento: {len(train_dataset)}")
print(f"Tamaño del conjunto de validación: {len(val_dataset)}")
print(f"Tamaño del conjunto de prueba: {len(test_dataset)}")
print(f"Número de clases: {len(train_dataset.info['label'])}")

# Definir la arquitectura CNN
class DermaCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(DermaCNN, self).__init__()

        # Primera capa convolucional
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
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
model = DermaCNN(num_classes=7).to(device)
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
            labels = labels.squeeze().long()  # DermaMNIST labels need reshaping

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels, total_loss / len(data_loader)

# Función de entrenamiento
def train_model(model, train_loader, val_loader, num_epochs=25):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze().long()

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

        # Validación
        val_preds, val_labels, val_loss = evaluate_model(model, val_loader, device)
        val_acc = 100 * accuracy_score(val_labels, val_preds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        scheduler.step()

        print(f'Época [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)

    return train_losses, val_losses, train_accuracies, val_accuracies

# Entrenar el modelo
print("Iniciando entrenamiento...")
train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, num_epochs=25)

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
print("RESULTADOS FINALES - DERMAMNIST CNN")
print("="*60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("="*60)

# Reporte detallado de clasificación
print("\nReporte detallado de clasificación:")
class_names = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
               'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']
print(classification_report(test_labels, test_preds, target_names=class_names))

# Función para plotear resultados
def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot de pérdidas
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Pérdida durante el entrenamiento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.legend()
    ax1.grid(True)

    # Plot de precisión
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
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
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión - DermaMNIST')
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Mostrar gráficos
plot_training_history(train_losses, val_losses, train_accs, val_accs)
plot_confusion_matrix(test_labels, test_preds, class_names)

# Guardar el modelo
torch.save(model.state_dict(), 'dermamnist_cnn_model.pth')
print("\nModelo guardado como 'dermamnist_cnn_model.pth'")

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
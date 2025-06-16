"""
model.py - Definición de la arquitectura CNN para BLOODMNIST
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BloodMNISTCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(BloodMNISTCNN, self).__init__()

        # Primera capa convolucional
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Segunda capa convolucional
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Tercera capa convolucional
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Cuarta capa convolucional
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Capas de pooling (usando MaxPool2d)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Capa de pooling promedio adaptativo
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout para regularización
        self.dropout = nn.Dropout(0.5)

        # Capas fully connected
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Primer bloque conv + batch norm + relu + pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Segundo bloque conv + batch norm + relu + pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Tercer bloque conv + batch norm + relu + pooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Cuarto bloque conv + batch norm + relu + adaptive pooling
        x = self.adaptive_pool(F.relu(self.bn4(self.conv4(x))))

        # Flatten para las capas fully connected
        x = x.view(x.size(0), -1)

        # Capas fully connected con dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def get_model_info(self):
        """Retorna información sobre el modelo"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'architecture': str(self)
        }


def create_model(num_classes=8, device='cpu'):
    """Factory function para crear el modelo"""
    model = BloodMNISTCNN(num_classes=num_classes)
    model.to(device)
    return model


def load_model(model_path, num_classes=8, device='cpu'):
    """Carga un modelo pre-entrenado"""
    model = BloodMNISTCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def save_model(model, model_path):
    """Guarda el modelo"""
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en: {model_path}")


if __name__ == "__main__":
    # Prueba del modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device=device)

    # Mostrar información del modelo
    info = model.get_model_info()
    print(f"Total de parámetros: {info['total_params']:,}")
    print(f"Parámetros entrenables: {info['trainable_params']:,}")

    # Prueba con datos dummy
    dummy_input = torch.randn(1, 3, 28, 28).to(device)
    output = model(dummy_input)
    print(f"Forma de salida: {output.shape}")
    print("Modelo funcionando correctamente!")
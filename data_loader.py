"""
data_loader.py - Manejo y carga del dataset BLOODMNIST
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class BloodMNISTDataManager:
    def __init__(self, batch_size=64, num_workers=2):
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.class_names = ['basophil', 'eosinophil', 'erythroblast', 'ig',
        #                     'lymphocyte', 'monocyte', 'neutrophil', 'platelet']
        self.class_names = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

        # Transformaciones
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Para visualización (sin normalización)
        self.viz_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_data(self, download=True):
        """Carga el dataset BLOODMNIST"""
        print(f"Cargando dataset BLOODMNIST...")

        #Cargar datasets BLOODMNIST
        self.train_dataset = medmnist.BloodMNIST(split='train', transform=self.transform, download=download)
        self.val_dataset = medmnist.BloodMNIST(split='val', transform=self.transform, download=download)
        self.test_dataset = medmnist.BloodMNIST(split='test', transform=self.transform, download=download)

        # Crear DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        print(f"Dataset cargado exitosamente!")
        self.print_dataset_info()

        return self.train_loader, self.val_loader, self.test_loader

    def print_dataset_info(self):
        """Imprime información del dataset"""
        print("\n" + "=" * 50)
        print("INFORMACIÓN DEL DATASET BLOODMNIST")
        print("=" * 50)
        print(f"Conjunto de entrenamiento: {len(self.train_dataset)} muestras")
        print(f"Conjunto de validación: {len(self.val_dataset)} muestras")
        print(f"Conjunto de prueba: {len(self.test_dataset)} muestras")
        print(f"Total de muestras: {len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)}")
        print(f"Número de clases: {len(self.class_names)}")
        print(f"Clases: {', '.join(self.class_names)}")
        print(f"Tamaño de batch: {self.batch_size}")

        # Mostrar distribución de clases en entrenamiento
        train_labels = [self.train_dataset[i][1].item() for i in range(len(self.train_dataset))]
        class_counts = Counter(train_labels)

        print(f"\nDistribución de clases en entrenamiento:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}: {class_counts[i]} muestras")
        print("=" * 50)

    def visualize_samples(self, num_samples=16, split='train'):
        """Visualiza muestras del dataset"""
        if split == 'train' and self.train_dataset is not None:
            dataset = medmnist.BloodMNIST(split='train', transform=self.viz_transform, download=False)
        elif split == 'val' and self.val_dataset is not None:
            dataset = medmnist.BloodMNIST(split='val', transform=self.viz_transform, download=False)
        elif split == 'test' and self.test_dataset is not None:
            dataset = medmnist.BloodMNIST(split='test', transform=self.viz_transform, download=False)
        else:
            print("Dataset no cargado o split inválido")
            return

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(f'Muestras del conjunto {split.upper()}', fontsize=16)

        indices = np.random.choice(len(dataset), num_samples, replace=False)

        for i, idx in enumerate(indices):
            row = i // 4
            col = i % 4

            image, label = dataset[idx]
            image = image.permute(1, 2, 0)  # CHW -> HWC

            axes[row, col].imshow(image)
            axes[row, col].set_title(f'{self.class_names[label.item()]}')
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

    def get_class_distribution(self, split='train'):
        """Obtiene la distribución de clases para un split específico"""
        if split == 'train' and self.train_dataset is not None:
            labels = [self.train_dataset[i][1].item() for i in range(len(self.train_dataset))]
        elif split == 'val' and self.val_dataset is not None:
            labels = [self.val_dataset[i][1].item() for i in range(len(self.val_dataset))]
        elif split == 'test' and self.test_dataset is not None:
            labels = [self.test_dataset[i][1].item() for i in range(len(self.test_dataset))]
        else:
            print("Dataset no cargado o split inválido")
            return None

        return Counter(labels)

    def plot_class_distribution(self):
        """Grafica la distribución de clases"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        splits = ['train', 'val', 'test']
        datasets = [self.train_dataset, self.val_dataset, self.test_dataset]

        for i, (split, dataset) in enumerate(zip(splits, datasets)):
            if dataset is not None:
                distribution = self.get_class_distribution(split)
                classes = list(range(len(self.class_names)))
                counts = [distribution[j] for j in classes]

                axes[i].bar(classes, counts, color=plt.cm.Set3(np.linspace(0, 1, len(self.class_names))))
                axes[i].set_title(f'Distribución - {split.upper()}')
                axes[i].set_xlabel('Clases')
                axes[i].set_ylabel('Número de muestras')
                axes[i].set_xticks(classes)
                axes[i].set_xticklabels([name[:3] for name in self.class_names], rotation=45)

                # Añadir números en las barras
                for j, count in enumerate(counts):
                    axes[i].text(j, count + max(counts) * 0.01, str(count),
                                 ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()

    def get_single_batch(self, split='train'):
        """Obtiene un solo batch para inspección"""
        if split == 'train' and self.train_loader is not None:
            return next(iter(self.train_loader))
        elif split == 'val' and self.val_loader is not None:
            return next(iter(self.val_loader))
        elif split == 'test' and self.test_loader is not None:
            return next(iter(self.test_loader))
        else:
            print("DataLoader no disponible")
            return None


def quick_load_data(batch_size=64, download=True):
    """Función rápida para cargar datos"""
    data_manager = BloodMNISTDataManager(batch_size=batch_size)
    return data_manager.load_data(download=download)


if __name__ == "__main__":
    # Ejemplo de uso
    print("Cargando dataset BLOODMNIST...")

    # Crear manager de datos
    data_manager = BloodMNISTDataManager(batch_size=32)

    # Cargar datos
    train_loader, val_loader, test_loader = data_manager.load_data()

    # Visualizar muestras
    data_manager.visualize_samples(num_samples=16, split='train')

    # Mostrar distribución de clases
    data_manager.plot_class_distribution()

    # Inspeccionar un batch
    batch_data, batch_labels = data_manager.get_single_batch('train')
    print(f"\nForma del batch de datos: {batch_data.shape}")
    print(f"Forma del batch de etiquetas: {batch_labels.shape}")
    print(f"Rango de valores en datos: [{batch_data.min():.3f}, {batch_data.max():.3f}]")

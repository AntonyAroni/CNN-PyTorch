"""
trainer.py - Entrenamiento del modelo CNN para BLOODMNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime


class CNNTrainer:
    def __init__(self, model, device='cpu', save_dir='./checkpoints'):
        self.model = model
        self.device = device
        self.save_dir = save_dir

        # Crear directorio de guardado si no existe
        os.makedirs(save_dir, exist_ok=True)

        # Métricas de entrenamiento
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []

        # Configuración de entrenamiento
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def setup_training(self, learning_rate=0.001, weight_decay=1e-4,
                       scheduler_step=7, scheduler_gamma=0.1):
        """Configura los componentes de entrenamiento"""
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=scheduler_step,
            gamma=scheduler_gamma
        )

        print(f"Configuración de entrenamiento:")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Weight decay: {weight_decay}")
        print(f"  - Scheduler step: {scheduler_step}")
        print(f"  - Scheduler gamma: {scheduler_gamma}")

    def train_epoch(self, train_loader):
        """Entrena una época"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Entrenando', leave=False)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            target = target.squeeze().long()

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Estadísticas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Actualizar barra de progreso
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader):
        """Valida una época"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validando', leave=False)

            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                target = target.squeeze().long()

                outputs = self.model(data)
                loss = self.criterion(outputs, target)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, num_epochs=20,
              save_best=True, save_every=5):
        """Entrenamiento completo"""

        print(f"\nIniciando entrenamiento por {num_epochs} épocas...")
        print("=" * 60)

        # Timestamp para identificar el entrenamiento
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for epoch in range(num_epochs):
            print(f"\nÉpoca [{epoch + 1}/{num_epochs}]")
            print("-" * 40)

            # Entrenar
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validar
            val_loss, val_acc = self.validate_epoch(val_loader)

            # Actualizar scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Guardar métricas
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(current_lr)

            # Imprimir métricas
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")

            # Guardar mejor modelo
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                if save_best:
                    best_model_path = os.path.join(self.save_dir, f'best_model_{timestamp}.pth')
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"✓ Nuevo mejor modelo guardado: {best_model_path}")

            # Guardar checkpoint cada cierto número de épocas
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch + 1}_{timestamp}.pth')
                self.save_checkpoint(checkpoint_path, epoch, val_acc)

        print(f"\n{'=' * 60}")
        print(f"Entrenamiento completado!")
        print(f"Mejor validación: {self.best_val_acc:.2f}% en época {self.best_epoch}")

        # Guardar modelo final
        final_model_path = os.path.join(self.save_dir, f'final_model_{timestamp}.pth')
        torch.save(self.model.state_dict(), final_model_path)

        # Guardar métricas de entrenamiento
        self.save_training_metrics(timestamp)

        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies

    def save_checkpoint(self, filepath, epoch, val_acc):
        """Guarda un checkpoint completo"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates
        }
        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint guardado: {filepath}")

    def load_checkpoint(self, filepath):
        """Carga un checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restaurar métricas
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
        self.learning_rates = checkpoint['learning_rates']

        print(f"Checkpoint cargado desde: {filepath}")
        print(f"Última época: {checkpoint['epoch']}")
        print(f"Última precisión de validación: {checkpoint['val_acc']:.2f}%")

        return checkpoint['epoch']

    def save_training_metrics(self, timestamp):
        """Guarda las métricas de entrenamiento en JSON"""
        metrics = {
            'timestamp': timestamp,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }

        metrics_path = os.path.join(self.save_dir, f'training_metrics_{timestamp}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Métricas guardadas: {metrics_path}")

    def plot_training_history(self, save_plot=True):
        """Visualiza el historial de entrenamiento"""
        if not self.train_losses:
            print("No hay métricas de entrenamiento para mostrar")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.train_losses) + 1)

        # Plot de pérdidas
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot de precisión
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot de learning rate
        ax3.plot(epochs, self.learning_rates, 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        # Plot de diferencia entre train y val accuracy
        acc_diff = [t - v for t, v in zip(self.train_accuracies, self.val_accuracies)]
        ax4.plot(epochs, acc_diff, 'm-', linewidth=2)
        ax4.set_title('Overfitting Check (Train - Val Accuracy)', fontsize=14)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy Difference (%)')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.save_dir, f'training_history_{timestamp}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico guardado: {plot_path}")

        plt.show()


def quick_train(model, train_loader, val_loader, device='cpu',
                num_epochs=20, learning_rate=0.001, save_dir='./checkpoints'):
    """Función rápida para entrenar un modelo"""
    trainer = CNNTrainer(model, device=device, save_dir=save_dir)
    trainer.setup_training(learning_rate=learning_rate)

    # Entrenar
    metrics = trainer.train(train_loader, val_loader, num_epochs=num_epochs)

    # Mostrar gráficos
    trainer.plot_training_history()

    return trainer, metrics


if __name__ == "__main__":
    # Ejemplo de uso
    from model import create_model
    from data_loader import quick_load_data

    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Cargar datos
    train_loader, val_loader, test_loader = quick_load_data(batch_size=64)

    # Crear modelo
    model = create_model(device=device)

    # Entrenar
    trainer, metrics = quick_train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=10,
        learning_rate=0.001
    )



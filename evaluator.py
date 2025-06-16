"""
evaluator.py - Evaluación y pruebas del modelo CNN para BLOODMNIST
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from tqdm import tqdm
import os


class ModelEvaluator:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.class_names = ['basophil', 'eosinophil', 'erythroblast', 'ig',
                            'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

    def evaluate(self, test_loader, verbose=True):
        """Evaluación completa del modelo"""
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_probabilities = []
        test_loss = 0.0
        correct = 0
        total = 0

        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluando') if verbose else test_loader

            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                target = target.squeeze().long()

                outputs = self.model(data)
                loss = criterion(outputs, target)

                # Obtener probabilidades
                probabilities = F.softmax(outputs, dim=1)

                # Obtener predicciones
                _, predicted = outputs.max(1)

                # Acumular resultados
                test_loss += loss.item()
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                if verbose:
                    pbar.set_postfix({'Acc': f'{100. * correct / total:.2f}%'})

        test_loss /= len(test_loader)
        test_acc = 100. * correct / total

        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probabilities)
        }

        if verbose:
            print(f"\nResultados de evaluación:")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.2f}%")

        return results

    def detailed_classification_report(self, targets, predictions):
        """Genera un reporte detallado de clasificación"""
        # Reporte de sklearn
        report = classification_report(
            targets, predictions,
            target_names=self.class_names,
            digits=4,
            output_dict=True
        )

        print("\n" + "=" * 80)
        print("REPORTE DETALLADO DE CLASIFICACIÓN")
        print("=" * 80)

        # Imprimir por clase
        print(f"{'Clase':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
        print("-" * 60)

        for i, class_name in enumerate(self.class_names):
            class_report = report[class_name]
            print(f"{class_name:<12} {class_report['precision']:<10.4f} "
                  f"{class_report['recall']:<10.4f} {class_report['f1-score']:<10.4f} "
                  f"{int(class_report['support']):<8}")

        print("-" * 60)
        print(f"{'Accuracy':<12} {'':<10} {'':<10} {report['accuracy']:<10.4f} "
              f"{int(report['macro avg']['support']):<8}")
        print(f"{'Macro Avg':<12} {report['macro avg']['precision']:<10.4f} "
              f"{report['macro avg']['recall']:<10.4f} {report['macro avg']['f1-score']:<10.4f} "
              f"{int(report['macro avg']['support']):<8}")
        print(f"{'Weighted Avg':<12} {report['weighted avg']['precision']:<10.4f} "
              f"{report['weighted avg']['recall']:<10.4f} {report['weighted avg']['f1-score']:<10.4f} "
              f"{int(report['weighted avg']['support']):<8}")

        return report

    def plot_confusion_matrix(self, targets, predictions, normalize=False, save_path=None):
        """Grafica la matriz de confusión"""
        cm = confusion_matrix(targets, predictions)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Matriz de Confusión Normalizada'
            fmt = '.2f'
        else:
            title = 'Matriz de Confusión'
            fmt = 'd'

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(title, fontsize=16)
        plt.xlabel('Predicción', fontsize=12)
        plt.ylabel('Valor Real', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Matriz de confusión guardada: {save_path}")

        plt.show()

        return cm

    def plot_class_accuracies(self, targets, predictions, save_path=None):
        """Grafica las precisiones por clase"""
        # Calcular precisión por clase
        class_accuracies = []
        class_counts = []

        for i in range(len(self.class_names)):
            mask = targets == i
            if mask.sum() > 0:
                acc = (predictions[mask] == i).sum() / mask.sum()
                class_accuracies.append(acc * 100)
                class_counts.append(mask.sum())
            else:
                class_accuracies.append(0)
                class_counts.append(0)

        # Crear gráfico
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Precisión por clase
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        bars1 = ax1.bar(range(len(self.class_names)), class_accuracies, color=colors)
        ax1.set_title('Precisión por Clase', fontsize=14)
        ax1.set_xlabel('Clases')
        ax1.set_ylabel('Precisión (%)')
        ax1.set_xticks(range(len(self.class_names)))
        ax1.set_xticklabels([name[:3] for name in self.class_names], rotation=45)
        ax1.set_ylim([0, 105])

        # Añadir valores en las barras
        for bar, acc in zip(bars1, class_accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

        # Número de muestras por clase
        bars2 = ax2.bar(range(len(self.class_names)), class_counts, color=colors)
        ax2.set_title('Número de Muestras por Clase (Test)', fontsize=14)
        ax2.set_xlabel('Clases')
        ax2.set_ylabel('Número de muestras')
        ax2.set_xticks(range(len(self.class_names)))
        ax2.set_xticklabels([name[:3] for name in self.class_names], rotation=45)

        # Añadir valores en las barras
        for bar, count in zip(bars2, class_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + max(class_counts) * 0.01,
                     str(count), ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico de precisiones guardado: {save_path}")

        plt.show()

        return class_accuracies, class_counts

    def plot_prediction_confidence(self, probabilities, targets, predictions, save_path=None):
        """Grafica la distribución de confianza en las predicciones"""
        # Calcular confianza (probabilidad máxima)
        confidences = np.max(probabilities, axis=1)
        correct_mask = predictions == targets

        plt.figure(figsize=(12, 5))

        # Subplot 1: Histograma de confianza
        plt.subplot(1, 2, 1)
        plt.hist(confidences[correct_mask], bins=30, alpha=0.7,
                 label='Predicciones Correctas', color='green', density=True)
        plt.hist(confidences[~correct_mask], bins=30, alpha=0.7,
                 label='Predicciones Incorrectas', color='red', density=True)
        plt.xlabel('Confianza de Predicción')
        plt.ylabel('Densidad')
        plt.title('Distribución de Confianza')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Precisión vs Confianza
        plt.subplot(1, 2, 2)
        confidence_bins = np.arange(0, 1.01, 0.1)
        bin_accuracies = []
        bin_counts = []

        for i in range(len(confidence_bins) - 1):
            mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i + 1])
            if mask.sum() > 0:
                acc = correct_mask[mask].mean()
                bin_accuracies.append(acc)
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)

        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        plt.plot(bin_centers, bin_accuracies, 'bo-', linewidth=2, markersize=6)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Calibración Perfecta')
        plt.xlabel('Confianza de Predicción')
        plt.ylabel('Precisión')
        plt.title('Curva de Calibración')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico de confianza guardado: {save_path}")

        plt.show()

        return confidences, bin_accuracies

    def analyze_errors(self, targets, predictions, probabilities, num_examples=5):
        """Analiza los errores más comunes"""
        incorrect_mask = predictions != targets
        incorrect_indices = np.where(incorrect_mask)[0]

        if len(incorrect_indices) == 0:
            print("¡No hay errores de clasificación!")
            return

        # Obtener confianzas de predicciones incorrectas
        incorrect_confidences = np.max(probabilities[incorrect_indices], axis=1)

        # Ordenar por confianza (errores más confiados primero)
        sorted_indices = incorrect_indices[np.argsort(-incorrect_confidences)]

        print(f"\n{'=' * 80}")
        print("ANÁLISIS DE ERRORES")
        print(f"{'=' * 80}")
        print(f"Total de errores: {len(incorrect_indices)} / {len(targets)} "
              f"({100 * len(incorrect_indices) / len(targets):.2f}%)")

        print(f"\nTop {min(num_examples, len(sorted_indices))} errores más confiados:")
        print(f"{'Idx':<6} {'Real':<12} {'Predicho':<12} {'Confianza':<10}")
        print("-" * 50)

        for i, idx in enumerate(sorted_indices[:num_examples]):
            real_class = self.class_names[targets[idx]]
            pred_class = self.class_names[predictions[idx]]
            confidence = np.max(probabilities[idx])

            print(f"{idx:<6} {real_class:<12} {pred_class:<12} {confidence:<10.4f}")

        return sorted_indices[:num_examples]

    def full_evaluation(self, test_loader, save_dir=None):
        """Evaluación completa con todos los análisis"""
        print("Iniciando evaluación completa...")

        # Evaluación básica
        results = self.evaluate(test_loader)

        targets = results['targets']
        predictions = results['predictions']
        probabilities = results['probabilities']

        # Reporte detallado
        report = self.detailed_classification_report(targets, predictions)

        # Matriz de confusión
        print("\nGenerando matriz de confusión...")
        cm_path = os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
        cm = self.plot_confusion_matrix(targets, predictions, save_path=cm_path)

        # Precisiones por clase
        print("\nAnalizando precisiones por clase...")
        acc_path = os.path.join(save_dir, 'class_accuracies.png') if save_dir else None
        class_accs, class_counts = self.plot_class_accuracies(targets, predictions, save_path=acc_path)

        # Análisis de confianza
        print("\nAnalizando confianza de predicciones...")
        conf_path = os.path.join(save_dir, 'prediction_confidence.png') if save_dir else None
        confidences, calibration = self.plot_prediction_confidence(
            probabilities, targets, predictions, save_path=conf_path
        )

        # Análisis de errores
        print("\nAnalizando errores...")
        error_indices = self.analyze_errors(targets, predictions, probabilities)

        # Compilar resultados completos
        full_results = {
            **results,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_accuracies': class_accs,
            'class_counts': class_counts,
            'confidences': confidences,
            'calibration': calibration,
            'error_indices': error_indices
        }

        return full_results


def quick_evaluate(model, test_loader, device='cpu', save_dir=None):
    """Función rápida para evaluar un modelo"""
    evaluator = ModelEvaluator(model, device=device)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    return evaluator.full_evaluation(test_loader, save_dir=save_dir)


if __name__ == "__main__":
    # Ejemplo de uso
    from model import load_model
    from data_loader import quick_load_data

    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Cargar datos
    _, _, test_loader = quick_load_data(batch_size=64)

    # Cargar modelo pre-entrenado (ajustar ruta)
    model_path = './checkpoints/best_model.pth'  # Cambiar por ruta real

    try:
        model = load_model(model_path, device=device)
        print(f"Modelo cargado desde: {model_path}")

        # Evaluar
        results = quick_evaluate(model, test_loader, device=device, save_dir='./evaluation_results')

    except FileNotFoundError:
        print(f"No se encontró el modelo en: {model_path}")
        print("Primero entrena un modelo usando trainer.py")
"""
main.py - Script principal para ejecutar entrenamiento y evaluación completa
source .venv/bin/activate

python main.py --mode evaluate --model-path ./checkpoints/trained_model_xxxx.pth

python main.py --mode both --epochs 15 --batch-size 32 --visualize-data

python main.py --mode evaluate --visualize-data --model-path ./checkpoints/best_model_best_model_num.pth
python main.py --mode evaluate --model-path ./checkpoints/best_model_20250619_192734.pth


"""


import torch
import argparse
import os
from datetime import datetime

# Importar nuestros módulos
from model import create_model, load_model, save_model
from data_loader import BloodMNISTDataManager
from trainer import CNNTrainer
from evaluator import ModelEvaluator


def setup_args():
    """Configurar argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='CNN para BLOODMNIST')

    # Argumentos generales
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'both'],
                        default='both', help='Modo de ejecución')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Tamaño de batch')
    parser.add_argument('--device', type=str, default='auto',
                        help='Dispositivo (cpu, cuda, auto)')

    # Argumentos de entrenamiento
    parser.add_argument('--epochs', type=int, default=20,
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--scheduler-step', type=int, default=7,
                        help='Step para scheduler')
    parser.add_argument('--scheduler-gamma', type=float, default=0.1,
                        help='Gamma para scheduler')

    # Argumentos de guardado/carga
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directorio para guardar modelos')
    parser.add_argument('--eval-dir', type=str, default='./evaluation_results',
                        help='Directorio para guardar resultados de evaluación')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Ruta del modelo pre-entrenado para evaluar')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Ruta del checkpoint para continuar entrenamiento')

    # Otros argumentos
    parser.add_argument('--no-download', action='store_true',
                        help='No descargar dataset (usar datos locales)')
    parser.add_argument('--visualize-data', action='store_true',
                        help='Visualizar muestras del dataset')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Guardar checkpoint cada N épocas')

    return parser.parse_args()


def setup_device(device_arg):
    """Configurar dispositivo de ejecución"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    print(f"Usando dispositivo: {device}")
    if device.type == 'cuda':
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return device


def load_data(args):
    """Cargar y preparar datos"""
    print("Configurando datos...")
    data_manager = BloodMNISTDataManager(batch_size=args.batch_size)

    # Cargar datos
    train_loader, val_loader, test_loader = data_manager.load_data(
        download=not args.no_download
    )

    # Visualizar datos si se solicita
    if args.visualize_data:
        print("Visualizando muestras del dataset...")
        data_manager.visualize_samples(num_samples=16, split='train')
        data_manager.plot_class_distribution()

    return train_loader, val_loader, test_loader, data_manager


def train_model(args, train_loader, val_loader, device):
    """Entrenar el modelo"""
    print(f"\n{'=' * 60}")
    print("INICIANDO ENTRENAMIENTO")
    print(f"{'=' * 60}")

    # Crear modelo
    model = create_model(device=device)

    # Mostrar información del modelo
    info = model.get_model_info()
    print(f"Total de parámetros: {info['total_params']:,}")
    print(f"Parámetros entrenables: {info['trainable_params']:,}")

    # Crear trainer
    trainer = CNNTrainer(model, device=device, save_dir=args.save_dir)

    # Configurar entrenamiento
    trainer.setup_training(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma
    )

    # Cargar checkpoint si se especifica
    start_epoch = 0
    if args.load_checkpoint:
        print(f"Cargando checkpoint: {args.load_checkpoint}")
        start_epoch = trainer.load_checkpoint(args.load_checkpoint)

    # Entrenar
    metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_every=args.save_every
    )

    # Mostrar gráficos de entrenamiento
    trainer.plot_training_history(save_plot=True)

    # Guardar modelo final
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(args.save_dir, f'trained_model_{timestamp}.pth')
    save_model(model, final_model_path)

    return model, trainer, final_model_path


def evaluate_model(args, test_loader, device, model_path=None, model=None):
    """Evaluar el modelo"""
    print(f"\n{'=' * 60}")
    print("INICIANDO EVALUACIÓN")
    print(f"{'=' * 60}")

    # Cargar modelo si no se proporciona
    if model is None:
        if model_path is None:
            raise ValueError("Debe proporcionar model_path o model")

        print(f"Cargando modelo desde: {model_path}")
        model = load_model(model_path, device=device)

    # Crear evaluador
    evaluator = ModelEvaluator(model, device=device)

    # Crear directorio de evaluación
    os.makedirs(args.eval_dir, exist_ok=True)

    # Evaluar
    results = evaluator.full_evaluation(test_loader, save_dir=args.eval_dir)

    print(f"\n{'=' * 60}")
    print("EVALUACIÓN COMPLETADA")
    print(f"{'=' * 60}")
    print(f"Precisión final: {results['test_accuracy']:.2f}%")
    print(f"Resultados guardados en: {args.eval_dir}")

    return results


def main():
    """Función principal"""
    # Configurar argumentos
    args = setup_args()

    # Configurar dispositivo
    device = setup_device(args.device)

    # Cargar datos
    train_loader, val_loader, test_loader, data_manager = load_data(args)

    # Variables para almacenar resultados
    model = None
    trainer = None
    model_path = args.model_path

    # Ejecutar según el modo
    if args.mode in ['train', 'both']:
        model, trainer, trained_model_path = train_model(args, train_loader, val_loader, device)
        if model_path is None:  # Si no se especificó modelo para evaluar, usar el recién entrenado
            model_path = trained_model_path

    if args.mode in ['evaluate', 'both']:
        results = evaluate_model(args, test_loader, device, model_path, model)

    print(f"\n{'=' * 60}")
    print("PROCESO COMPLETADO")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
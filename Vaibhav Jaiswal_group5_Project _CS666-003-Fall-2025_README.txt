Knee Osteoarthritis (KOA) Classification using Hybrid Deep Learning Models: ResNet50 & YOLOv8
Project Overview

This project combines two deep learning models, ResNet50 and YOLOv8, to classify the stages of Knee Osteoarthritis (KOA) using X-ray images. The hybrid model improves early-stage KOA detection (Grade 1 and Grade 2), a key challenge in the field. YOLOv8 is used for region localization, while ResNet50 performs feature extraction and classification. The model is designed for real-time clinical use and offers improvements in both accuracy and efficiency.

Requirements
Software Dependencies

Python 3.x - Python 3.7 or higher is required.

PyTorch 1.9+ - Essential for implementing deep learning models.

YOLOv8 - For object localization and classification of KOA stages.

torchvision - Used for image transformations and pre-trained models.

matplotlib - Required for plotting training curves and visualizing results.

scikit-learn - For generating classification metrics and confusion matrices.

Hardware Requirements

GPU (CUDA-enabled) - Needed for efficient model training and inference.

RAM - At least 8GB of RAM is recommended for optimal performance.

Disk Space - Ensure sufficient space for storing the dataset (approximately 2-3 GB for KOA dataset).

Setup Instructions

Clone the repository or download the project files.

Ensure you have YOLOv8 and ResNet50 pre-trained models available.

Modify the DATA_ROOT path in both the ResNet50 and YOLOv8 sections to point to your local dataset directory.

Install required libraries using the following command:

pip install torch torchvision matplotlib scikit-learn ultralytics

How to Run the Code
1. Train the ResNet50 Model

Train a ResNet50 model for KOA classification.

# ---------------- CONFIG (edit paths as needed) ----------------
TRAIN_DIR = r"C:\path_to_train_dataset"
VAL_DIR   = r"C:\path_to_val_dataset"
TEST_DIR  = r"C:\path_to_test_dataset"

IMG_SIZE     = 224
BATCH_SIZE   = 32        
NUM_WORKERS  = 4         
EPOCHS       = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 6
SAVE_PATH    = "kneexray_resnet50_best.pt"


Run ResNet50 training by modifying dataset paths.

The model will train for 30 epochs using AdamW optimizer with CosineAnnealingLR scheduler.

Early stopping is enabled after 6 epochs with no improvement in validation accuracy.

2. Train the YOLOv8 Model

Train a YOLOv8 model for KOA classification with the same dataset.

from ultralytics import YOLO

DATA_ROOT = r"C:\path_to_dataset"

MODEL = "yolov8m-cls.pt"  
DEVICE = 0  

model = YOLO(MODEL)

results = model.train(
    data=DATA_ROOT,
    epochs=30,
    imgsz=224,
    batch=64,
    device=DEVICE,
    patience=8,
    lr0=1e-3,
    weight_decay=5e-4,
    augment=True,
    mixup=0.1,
    cutmix=0.2,
    dropout=0.1,
    project="runs/classify",
    name="kneexray_yolov8m",
)


Train YOLOv8 using the dataset for KOA classification.

The model will train for 30 epochs with augmentation enabled (mixup, cutmix) to improve generalization.

The best weights will be saved during training.

Evaluation and Results

Once both models are trained, use the evaluation script to test the model on the test dataset:

# Load the best model weights and evaluate on the test set
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Evaluate using classification report and confusion matrix
evaluate(test_loader)


Expected Metrics:

Accuracy: The percentage of correctly classified KOA grades.

Precision, Recall, F1-Score: Key metrics for assessing the model’s ability to detect specific KOA grades.

Confusion Matrix: Visual representation of model performance.

Output

Training Curves: Graphs of accuracy and loss during training and validation.

Classification Report: Precision, recall, F1-score, and support for each KOA grade.

Confusion Matrix: Visual representation of model performance.

Next Steps

Fine-tuning: Explore hyperparameter tuning to optimize performance, particularly for early KOA detection.

Extend Dataset: Include more diverse examples for early-stage KOA (Grade 1 and Grade 2) to enhance model robustness.

Deploy: Investigate deploying the model in real-time clinical environments for automatic KOA grading from X-ray images.

Troubleshooting

Low GPU Memory: If out-of-memory errors occur, reduce the batch size or use smaller images.

Model Overfitting: Use data augmentation or add dropout layers to prevent overfitting and improve generalization.

Key Updates:

Added clarity on the hybrid model’s components (YOLOv8 for localization and ResNet50 for classification).

Updated sections for training YOLOv8 and ResNet50 models with detailed instructions.

Emphasized real-time clinical application and future improvements like data augmentation and fine-tuning.
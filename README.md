# 🧠 **Model Architecture — MLOps Gun Detection**

This branch represents the **Model Architecture stage** of the **MLOps Gun Detection** pipeline.
Here, the project evolves from pure data preparation into **deep learning model design**, defining the **Faster R-CNN (ResNet-50 FPN)** backbone, compilation logic, and structured training routine.

The focus of this stage is to **create a reusable and configurable object detection model**, integrate it with the preprocessed dataset, and establish a foundation for systematic model training in the next stage.

## 🧾 **What This Stage Includes**

* ✅ New module: `src/model_architecture.py` defining the `FasterRCNNModel` class
* ✅ Model creation using **Faster R-CNN with ResNet-50 FPN backbone**
* ✅ Customisable classifier head for multi-class detection
* ✅ Model compilation with **Adam optimiser** and configurable learning rate
* ✅ Structured training loop with **per-epoch progress and loss logging**
* ✅ Seamless integration with the preprocessed dataset (`GunDataset`)
* ✅ Comprehensive logging and error handling across all operations

This stage transforms the pipeline from **data preprocessing** into **model configuration and training orchestration**, enabling reproducible experimentation and modular model development.

## 🗂️ **Updated Project Structure**

```text
mlops-gun-detection/
├── artifacts/
├── src/
│   ├── custom_exception.py          # Unified error handling
│   ├── data_ingestion.py            # Dataset download and extraction
│   ├── data_processing.py           # GunDataset class for image + label preprocessing
│   ├── model_architecture.py        # Faster R-CNN definition and training logic
│   ├── logger.py                    # Centralised logging configuration
│   └── __init__.py
├── requirements.txt
├── setup.py
└── README.md                        # 📖 You are here
```

> 💡 The dataset remains sourced from [**issaisasank/guns-object-detection**](https://www.kaggle.com/datasets/issaisasank/guns-object-detection).
> GPU acceleration is strongly recommended for all training-related operations.
> Use the preprocessed dataset under `artifacts/raw/` as input for model development.

## 🧩 **Key Module Highlights**

### 🔹 `src/model_architecture.py` — Faster R-CNN Definition

Implements the **`FasterRCNNModel`** class, encapsulating all model-related operations:
creation, compilation, and training.

Handles:

* Model loading with **pretrained COCO weights (ResNet-50 FPN)**
* Custom classifier head for binary (gun vs background) detection
* Compilation using **Adam optimiser**
* Device management (automatic GPU/CPU assignment)
* Epoch-based training with **progress bars (TQDM)**
* Integrated logging and exception handling for every step

This class ensures that model training is reproducible, modular, and easy to monitor, forming the blueprint for the upcoming training pipeline.

### 🔹 `src/data_processing.py` — Dataset Interface

Defines the **`GunDataset`** class, responsible for converting raw image and label files into normalised PyTorch tensors.
This dataset feeds directly into the `train()` method of the model for supervised training.

### 🔹 `src/logger.py` — Logging System

Centralises all runtime logs from model creation, compilation, and training loops.
Each epoch and loss value is logged for traceability.

### 🔹 `src/custom_exception.py` — Exception Handling

Ensures consistent and clear exception messages when model creation, optimiser setup, or training steps fail.
Provides complete traceback information for debugging.

## ⚙️ **Testing the Model Module**

You can test the model creation and training process with:

```bash
python src/model_architecture.py
```

Or, use an interactive example:

```python
from src.model_architecture import FasterRCNNModel
from src.data_processing import GunDataset
from torch.utils.data import DataLoader
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare dataset and DataLoader
dataset = GunDataset(root="artifacts/raw", device=device)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# Build and train the model
model = FasterRCNNModel(num_classes=2, device=device)
model.compile(lr=1e-4)
model.train(train_loader, num_epochs=5)
```

Expected output (sample):

```
2025-10-21 14:12:03,110 - INFO - ✅ Model architecture initialised successfully.
2025-10-21 14:12:03,890 - INFO - ⚙️ Model compiled successfully with learning rate 0.0001
2025-10-21 14:12:04,015 - INFO - 🚀 Epoch 1 started...
2025-10-21 14:12:36,247 - INFO - ✅ Epoch 1 completed | Total Loss: 1.3478
```

This confirms that the model is correctly instantiated, compiled, and trained for the specified number of epochs.

## 🧠 **Outputs**

| Output                 | Description                                           |
| :--------------------- | :---------------------------------------------------- |
| **Model Object**       | Initialised Faster R-CNN model with modified head     |
| **Compiled Optimiser** | Adam optimiser with configured learning rate          |
| **Training Logs**      | Epoch-wise training progress and total loss           |
| **Error Handling**     | Contextualised exceptions for model setup or training |

## 🧩 **Integration with MLOps Pipeline**

This stage marks the transition into the **model layer** of the MLOps Gun Detection pipeline.
The model architecture defined here will be integrated with future components for automated training and deployment.

In the next stages:

* The **FasterRCNNModel** class will be integrated into `training_pipeline.py`.
* Training, validation, and checkpointing will be automated.
* Model weights will be versioned under `artifacts/models/`.
* Evaluation metrics (precision, recall, IoU) will be added for monitoring model performance.

This integration will ensure a seamless flow from **data → model → training → evaluation**.

## 🚀 **Next Stage — Model Training**

The next branch introduces the **Model Training stage**, focusing on structured experiment management and training automation.
This will include:

* Integration of the `FasterRCNNModel` into a reusable **training pipeline**
* Addition of **evaluation and checkpoint saving** logic
* Logging of key performance metrics per epoch
* Versioning of trained weights under `artifacts/models/`
* Preparation for deployment-ready inference pipelines

This marks the transition from **model design** to **systematic training and evaluation**, laying the groundwork for reproducible machine learning in production.

## ✅ **Best Practices**

* Always verify CUDA availability before model training.
* Keep the `num_classes` parameter consistent with your dataset.
* Monitor logs under `logs/` to track epoch progress and potential issues.
* Validate training stability with small batch sizes before scaling up.
* Use version control (e.g., Git branches) for each experimental architecture variant.

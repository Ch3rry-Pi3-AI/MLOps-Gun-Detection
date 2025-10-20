# 🏋️ **Model Training — MLOps Gun Detection**

This branch represents the **Model Training stage** of the **MLOps Gun Detection** pipeline.
Here, we move from model definition to a **reproducible training workflow** that handles dataset splitting, dataloaders, training/validation loops, TensorBoard logging, and checkpoint saving.

For fast iteration in this stage, training uses a **small subset of 5 samples**. This is intentional and will be expanded again in later stages.

## 🧾 **What This Stage Includes**

* ✅ New module: `src/model_training.py` with the `ModelTraining` class
* ✅ Automated **train/validation split** + `DataLoader` creation
* ✅ Epoch-based **training loop** with backprop and optimiser steps (Adam)
* ✅ Post-epoch **validation** and console logging
* ✅ **TensorBoard** logging for training loss (`tensorboard_logs/`)
* ✅ **Model checkpoint** saving to `artifacts/models/fasterrcnn.pth`
* ✅ Robust **logging** and **exception handling** throughout

This stage turns the architecture into a **working training system** that can be scaled in future branches.

## 🗂️ **Updated Project Structure**

```text
mlops-gun-detection/
├── artifacts/
│   └── models/                      # Saved model checkpoints
├── tensorboard_logs/                # TB runs created by ModelTraining
├── src/
│   ├── custom_exception.py          # Unified error handling
│   ├── data_ingestion.py            # Dataset download and extraction
│   ├── data_processing.py           # GunDataset class for image + label preprocessing
│   ├── logger.py                    # Centralised logging configuration
│   ├── model_architecture.py        # Faster R-CNN definition and compile/train helpers
│   └── model_training.py            # Training orchestration, validation, checkpoints, TB
├── requirements.txt
├── setup.py
└── README.md                        # 📖 You are here
```

> 💡 In this branch, `ModelTraining.split_dataset()` limits the dataset to a **subset of 5** for quick experimentation. This cap will be removed in later stages to train on the full dataset.

## 🧩 **Key Module Highlights**

### 🔹 `src/model_training.py` — Training Orchestration

Implements the **`ModelTraining`** class to coordinate:

* Train/val split and `DataLoader` creation
* Training loop (loss forward pass → backward → optimiser step)
* Validation loop after each epoch
* **TensorBoard** scalar logging: `Loss/train`
* Checkpoint saving to `artifacts/models/fasterrcnn.pth`

### 🔹 `src/model_architecture.py` — Model Builder

Creates and compiles **Faster R-CNN (ResNet-50 FPN)** and exposes a clean API the trainer can use.

### 🔹 `src/data_processing.py` — Dataset Interface

Supplies normalised tensors and target dictionaries compatible with TorchVision’s detection models.

## ⚙️ **Running Training**

```bash
python src/model_training.py
```

Or, instantiate programmatically:

```python
from src.model_training import ModelTraining
from src.model_architecture import FasterRCNNModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

trainer = ModelTraining(
    model_class=FasterRCNNModel,
    num_classes=2,
    learning_rate=1e-4,
    epochs=1,
    dataset_path="artifacts/raw/",
    device=device
)
trainer.train()
```

Expected console logs (sample):

```
✅ Dataset successfully split into training and validation sets.
🚀 Starting Epoch 1/1
📉 Validation Loss Type: <class 'dict'>
VAL_LOSS: {'loss_classifier': tensor(...), 'loss_box_reg': tensor(...), ...}
✅ Model saved successfully at artifacts/models/fasterrcnn.pth
```

## 🧠 **Outputs**

| Output                      | Description                                                |
| :-------------------------- | :--------------------------------------------------------- |
| **Model Checkpoint**        | `artifacts/models/fasterrcnn.pth`                          |
| **TensorBoard Logs**        | Scalars written under `tensorboard_logs/`                  |
| **Training Console Logs**   | Epoch progress and validation summaries                    |
| **Exception-safe Workflow** | Consistent error messages via `CustomException` + `logger` |

## 🧩 **Integration with MLOps Pipeline**

This training stage integrates seamlessly with the existing modules:

* Uses `GunDataset` to load preprocessed samples
* Consumes the `FasterRCNNModel` API for model creation/compilation
* Produces **checkpoints** and **logs** for downstream evaluation and deployment

Upcoming enhancements will expand dataset size, add proper validation metrics (mAP/IoU), and introduce experiment management.

## 🚀 **Next Stage — Experiment Tracking with TensorBoard**

The next branch focuses on **Experiment Tracking**:

* Standardise **TensorBoard** logging across metrics (loss components, learning rate, epoch timings)
* Add **run metadata** (hyperparameters, dataset size, seed)
* Provide **how-to** docs for launching and comparing runs (e.g., `tensorboard --logdir tensorboard_logs`)
* Prepare structure for future integration with MLflow/W&B if desired

## ✅ **Best Practices**

* Start with small epochs and batch sizes; scale up after stability checks
* Always verify CUDA availability (`torch.cuda.is_available()`)
* Version your checkpoints under `artifacts/models/`
* Keep logs tidy: one run directory per experiment in `tensorboard_logs/`
* Record your training config in git (commit the exact code you ran)
# 🧠 **Data Processing & Dataset Preparation — MLOps Gun Detection**

This branch represents the **data preparation stage** of the **MLOps Gun Detection** pipeline.
Here, the project evolves from experimental Kaggle notebook development into a **modular, reusable data processing system**.

The focus of this stage is to **standardise dataset handling**, build a **robust PyTorch Dataset class**, and ensure that image–label pairs are properly loaded, validated, and moved to the correct device (CPU/GPU) with full logging and error handling support.

## 🧾 **What This Stage Includes**

* ✅ New module: `src/data_processing.py` implementing the `GunDataset` class
* ✅ Integration of **custom logging** and **exception handling** for all preprocessing steps
* ✅ Normalisation and tensor conversion of image data using **OpenCV** and **PyTorch**
* ✅ Automatic computation of **bounding box areas** and **object labels**
* ✅ Robust file verification and error management with `CustomException`
* ✅ Standalone test block to validate dataset integrity

This stage transforms the project from **notebook-based experimentation** into a **structured and traceable data preparation layer**, forming the bridge between dataset ingestion and model training.

## 🗂️ **Updated Project Structure**

```text
mlops-gun-detection/
├── artifacts/
│   └── raw/
│       ├── Images/                  # Image samples from Kaggle dataset
│       └── Labels/                  # Text files containing bounding box coordinates
├── src/
│   ├── custom_exception.py          # Unified error handling
│   ├── data_ingestion.py            # Dataset download and extraction
│   ├── data_processing.py           # New GunDataset class for image + label loading
│   ├── logger.py                    # Centralised logging configuration
│   └── __init__.py
├── requirements.txt
├── setup.py
└── README.md                        # 📖 You are here
```

> 💡 The dataset for this stage remains sourced from **Kaggle**, under
> [https://www.kaggle.com/datasets/issaisasank/guns-object-detection](https://www.kaggle.com/datasets/issaisasank/guns-object-detection).
> Ensure GPU acceleration is available if you plan to verify data loading in a Kaggle or local CUDA-enabled environment.

## 🧩 **Key Module Highlights**

### 🔹 `src/data_processing.py` — Dataset Definition

Implements the **`GunDataset`** class that prepares the dataset for object detection training.
Handles:

* Image reading via OpenCV
* RGB conversion and normalisation
* Bounding box loading from text files
* Label and area computation
* Device transfer and tensor formatting
* Detailed logging and structured exception handling

This class ensures every data sample is validated, logged, and returned in the exact structure expected by **Faster R-CNN** and other object detection models.

### 🔹 `src/logger.py` — Logging System

Handles all structured logging within the pipeline.
Each operation (image load, label parse, dataset check) is timestamped and written to a daily log file under `logs/`.

### 🔹 `src/custom_exception.py` — Exception Handling

Guarantees consistent and traceable error reporting across modules, identifying both the source file and line number of failures.

### 🔹 `src/data_ingestion.py` — Dataset Retrieval

Maintains the ingestion logic for downloading and preparing the Kaggle dataset, ensuring the correct directory structure under `artifacts/raw/`.

## ⚙️ **Testing the Module**

To validate dataset loading and verify proper image–label pairing, you can run:

```bash
python src/data_processing.py
```

Expected output (sample):

```
2025-10-20 19:05:42,018 - INFO - ✅ Data Processing Initialised...
2025-10-20 19:05:42,152 - INFO - 📸 Loading data for index 0
2025-10-20 19:05:42,384 - INFO - Image Path: artifacts/raw/Images/001.jpeg
🖼️ Image Shape: torch.Size([3, 480, 640])
📦 Target Keys: dict_keys(['boxes', 'area', 'image_id', 'labels'])
🔲 Bounding Boxes: tensor([[ 54., 129., 212., 320.]])
```

This confirms that images and bounding boxes are loaded correctly and tensors are transferred to the specified device.

## 🧠 **Outputs**

| Output                   | Description                                               |
| :----------------------- | :-------------------------------------------------------- |
| **Preprocessed Dataset** | Images and bounding boxes converted into PyTorch tensors  |
| **Target Dictionary**    | Contains bounding boxes, labels, areas, and image IDs     |
| **Logging Output**       | Detailed logs for dataset loading and preprocessing       |
| **Custom Exceptions**    | Traceable, formatted error messages during I/O operations |

## 🧩 **Integration with MLOps Pipeline**

This stage formalises the **data foundation** for downstream model development and training.
In the next stages:

* The **GunDataset** class will feed directly into the model training pipeline.
* Integration with `training_pipeline.py` will enable reproducible dataset handling during training and validation.
* Dataset transformations and augmentations can be modularised into a dedicated **preprocessing pipeline**.
* All data loading and processing events will remain fully logged and auditable.

## 🚀 **Next Stage — Model Architecture**

The next branch introduces the **Model Architecture** stage, where the Faster R-CNN design is defined and integrated with the preprocessed dataset.
This will include:

* A dedicated `src/model_architecture.py` module for building and configuring Faster R-CNN.
* Integration of transfer learning from pre-trained COCO weights.
* Compilation methods for optimiser, learning rate, and scheduler setup.
* Training and evaluation hooks for modular reuse within `training_pipeline.py`.

This marks the transition from **data preparation** to **deep learning model design and configuration**, paving the way for structured, production-ready training pipelines.

## ✅ **Best Practices**

* Use GPU-enabled environments for validating large datasets.
* Keep `logger` calls concise but informative for debugging.
* Avoid hard-coded paths — reference directories dynamically using `config/paths_config.py`.
* Test the dataset locally before running large training jobs.
* Document changes and test outputs to maintain full reproducibility.
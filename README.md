# 🎯 **Exploratory Training & Inference — MLOps Gun Detection**

This branch represents the **data scientist’s experimental stage**, where the **Guns Object Detection dataset** (sourced from Kaggle) is explored, visualised, and used to train a **Faster R-CNN** model for firearm detection.

The goal of this stage is to **understand dataset structure**, perform **data loading and visualisation experiments**, and develop a **baseline object detection model** — before the workflow is modularised into automated preprocessing and training pipelines.

## 🧾 **What This Stage Includes**

* ✅ Jupyter Notebook (`notebook/guns-object-detection.ipynb`) for model experimentation on **Kaggle**
* ✅ Dataset source: [**issaisasank/guns-object-detection**](https://www.kaggle.com/datasets/issaisasank/guns-object-detection)
* ✅ Custom **PyTorch Dataset** for image–label pairing
* ✅ **Data visualisation** utilities to display bounding boxes over images
* ✅ Implementation of **Faster R-CNN (ResNet-50 FPN)** model for object detection
* ✅ Training loop with **Adam optimiser** and per-epoch loss reporting
* ✅ Application of **Non-Maximum Suppression (NMS)** for post-processing predictions
* ✅ Visualisation of **ground-truth vs predicted** bounding boxes
* ✅ Baseline experimentation environment — fully GPU-accelerated on **Kaggle**

This notebook functions as a **sandbox for the data scientist**, allowing iterative experimentation before converting the logic into reusable preprocessing and training modules for production.

## 🗂️ **Updated Project Structure**

```
mlops-gun-detection/
├── artifacts/
│   └── raw/
│       ├── Images/                  # Image samples from Kaggle dataset
│       └── Labels/                  # Text files containing bounding box coordinates
├── notebook/
│   └── guns-object-detection.ipynb  # 🔍 Model training and visualisation notebook
├── config/
├── src/
├── requirements.txt
├── setup.py
└── README.md                        # 📖 You are here
```

> 💡 The dataset used in this stage is automatically available within the **Kaggle notebook environment** under `/kaggle/input/guns-object-detection/`.
> GPU acceleration (P100) must be enabled via **Settings → Accelerator → GPU (P100)**.
> Users may need to **sign up and verify** their Kaggle account to enable GPU access.

## 🧩 **Notebook Highlights**

Within `notebook/guns-object-detection.ipynb`, you’ll find the following structured sections:

1. **Setup & Device Configuration** — imports dependencies, detects CUDA availability, and prepares the environment.

2. **Data Loading & Visualisation** — defines helper functions to display annotated images with bounding boxes.

3. **Dataset Preparation** — builds a custom `Gun_Data` PyTorch Dataset class to handle image–label pairing.

4. **Model Definition** — initialises a pretrained **Faster R-CNN (ResNet-50 FPN)** with custom output classes.

5. **Training Loop** — compiles the model, configures the optimiser, and trains over 30 epochs.

6. **Inference & Post-Processing** — runs predictions, applies **Non-Maximum Suppression**, and prepares results.

7. **Visualisation** — displays both **ground-truth** and **predicted** bounding boxes using OpenCV and Matplotlib.

## ⚙️ **Running the Notebook**

Because this notebook depends on GPU acceleration and Kaggle’s pre-mounted dataset structure, it must be executed **directly on Kaggle**.

### Steps:

1. Go to the dataset page:
   👉 [https://www.kaggle.com/datasets/issaisasank/guns-object-detection](https://www.kaggle.com/datasets/issaisasank/guns-object-detection)

2. Click **`<> Code`** → **New Notebook**.

3. Open the right-hand **Settings** panel.

4. Under **Accelerator**, select **GPU (P100)**.

5. If GPU is greyed out, verify your Kaggle account via email or phone.

6. Copy or upload `guns-object-detection.ipynb` into the editor.

7. Run all cells sequentially to load data, train the model, and visualise predictions.

## 🧠 **Outputs**

| Output                        | Description                                             |
| :---------------------------- | :------------------------------------------------------ |
| **Trained Model (in-memory)** | Faster R-CNN model trained for firearm detection.       |
| **Predictions**               | Bounding boxes and confidence scores for detected guns. |
| **Visualisations**            | Ground-truth vs predicted box plots using Matplotlib.   |
| **Training Logs**             | Epoch-wise loss metrics displayed in notebook output.   |

## 🧩 **Integration with MLOps Pipeline**

This notebook establishes the foundation for a **reproducible MLOps pipeline** in subsequent branches:

* The **`Gun_Data`** dataset class and **`Model`** wrapper will be modularised under `src/data_processing.py` and `src/model_training.py`.

* Future pipelines will automate dataset ingestion, preprocessing, training, and inference tracking.

* Trained weights will be versioned and stored under `artifacts/models/` for downstream deployment.

* Integration with **Flask or FastAPI inference apps** will follow after preprocessing automation.

## 🚀 **Next Stage — Data Preprocessing**

The next branch evolves this experimental notebook into a structured **data preprocessing pipeline**, featuring:

* Creation of `src/data_processing.py` for automated normalisation, augmentation, and transformation logic.
* Parameter and path configuration updates under `config/`.
* Artefact storage under `artifacts/processed/` for versioned, reusable datasets.
* Integration of logging and exception handling to ensure pipeline robustness.

This transition marks the evolution from **GPU-based research** → **modular data preprocessing pipeline**, bridging the gap between **experimentation and engineered automation**.

## ✅ **Best Practices**

* Always enable **GPU (P100)** before training.
* Run cells sequentially from top to bottom — the workflow is linear.
* Save sample predictions and bounding box plots for reporting.
* Keep code modular and move reusable components into `src/` as the project matures.
* Treat this notebook as a **sandbox** — production logic will be implemented in future pipeline stages.
# 🔫 **Object Detection with Faster R-CNN — `guns-object-detection.ipynb`**

This notebook implements a **complete training and inference pipeline** for detecting firearms in images using a **Faster R-CNN** model with a **ResNet-50 backbone**.
It leverages the **Guns Object Detection Dataset** hosted on **Kaggle**, performing data loading, preprocessing, model definition, training, and visualisation — all in a single, GPU-enabled environment.

> ⚠️ **Note:** This notebook must be executed on **Kaggle**, not locally.
> Kaggle provides free access to GPUs required for Faster R-CNN training.

## 📁 **File Location**

```
mlops-gun-detection/
├── notebook/
│   └── guns-object-detection.ipynb
├── src/
├── config/
└── artifacts/
    └── raw/
        ├── Images/       # Input images
        └── Labels/       # Corresponding bounding box coordinates
```
## 🎯 **Purpose**

This notebook serves as a **research and experimentation environment** for:

1. Exploring the [**Guns Object Detection Dataset**](https://www.kaggle.com/datasets/issaisasank/guns-object-detection).
2. Loading and visualising annotated images with bounding boxes.
3. Building a custom **PyTorch Dataset** for object detection.
4. Implementing and training a **Faster R-CNN (ResNet-50 FPN)** model.
5. Applying **Non-Maximum Suppression (NMS)** to filter overlapping predictions.
6. Visualising both **ground-truth** and **predicted** bounding boxes.
7. Establishing baseline results for downstream model deployment.


## 🧭 **Before You Begin**

You’ll need to run this notebook **directly on Kaggle** to access the dataset and GPU resources.

### 🧩 **Steps:**

1. Visit the dataset page:
   👉 [https://www.kaggle.com/datasets/issaisasank/guns-object-detection](https://www.kaggle.com/datasets/issaisasank/guns-object-detection)

2. Click the **`<> Code`** tab.

3. Select **“New Notebook”**.

4. In the right-hand **Settings** panel:

   * Set **Accelerator → GPU (P100)**
   * (If unavailable, sign up and verify your Kaggle account — verification is sometimes required before GPU access.)

5. Copy the notebook code or upload this file into the editor.

6. Run all cells sequentially to:

   * Load data,
   * Train the model,
   * And visualise detection results.


## 🧩 **Structure Overview**

| Section                             | Description                                                                                       |
| :---------------------------------- | :------------------------------------------------------------------------------------------------ |
| **1. Setup**                        | Imports dependencies, configures the runtime device (CPU/GPU), and initialises the environment.   |
| **2. Data Loading & Visualisation** | Defines helper functions to visualise annotated images with bounding boxes.                       |
| **3. Dataset Preparation**          | Builds a custom PyTorch Dataset class to load and format images and labels for Faster R-CNN.      |
| **4. Model Definition & Training**  | Defines the Faster R-CNN model, attaches an optimiser, and trains for multiple epochs.            |
| **5. Inference & Evaluation**       | Runs predictions on sample images, applies Non-Maximum Suppression (NMS), and visualises results. |


## ⚙️ **Requirements**

All dependencies are pre-installed in the Kaggle environment.
No manual setup is required beyond enabling the GPU accelerator.

If you wish to run it locally (optional and not recommended), you can install the dependencies using:

```bash
pip install torch torchvision albumentations opencv-python matplotlib tqdm pillow
```

However, **training performance will be severely limited without a GPU**.

## 🧠 **Typical Workflow**

1. Load the Kaggle dataset under `/kaggle/input/guns-object-detection/`.
2. Inspect a few sample images with bounding boxes using the `imgshow()` helper.
3. Initialise the `Gun_Data` PyTorch Dataset class to load and preprocess images.
4. Create and compile a `Model` wrapper for Faster R-CNN with two classes: background and gun.
5. Split the dataset (80/20) and load it via `DataLoader` objects.
6. Train the model for 30 epochs using an **Adam optimiser**.
7. Run inference on unseen images and apply **NMS** to refine predictions.
8. Display visual comparisons between **ground-truth** and **predicted** bounding boxes.

## 🧾 **Outputs**

| Output                        | Description                                                            |
| :---------------------------- | :--------------------------------------------------------------------- |
| **Trained Model (in-memory)** | A Faster R-CNN model trained on the Guns Object Detection dataset.     |
| **Predictions**               | Bounding boxes and confidence scores for detected guns in test images. |
| **Visualisations**            | Plots comparing ground-truth boxes and model predictions post-NMS.     |
| **Training Logs**             | Per-epoch loss metrics printed during model training.                  |

## 🧩 **Integration with MLOps Pipeline**

This notebook acts as the **experimental foundation** for productionising object detection workflows:

* The `Gun_Data` dataset class and `Model` wrapper can later be modularised into `src/data_processing.py` and `src/model_training.py`.
* Trained weights can be exported under `artifacts/models/` for inference pipelines.
* Evaluation and bounding box visualisation logic can feed into a **Flask inference app** or **streamlit dashboard** for deployment.
* Future stages will include **model registry**, **continuous training**, and **inference serving** using the same architecture.

## ✅ **Best Practices**

* Always enable **GPU (P100)** before running training cells in Kaggle.
* Use small batch sizes (e.g. 2–3) to prevent GPU memory overflow.
* Run all cells sequentially from top to bottom — the workflow is linear.
* Save visual outputs or screenshots of predictions for documentation.
* Treat this notebook as a **sandbox** — reusable components should be migrated to Python modules in `src/`.
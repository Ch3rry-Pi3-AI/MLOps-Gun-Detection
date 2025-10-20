# 📊 **Experiment Tracking with TensorBoard — MLOps Gun Detection**

This branch is a quick, focused stage: we’ve already wired training to log **TensorBoard** scalars. All you need to do is launch TensorBoard and compare runs.

## ⚙️ Run TensorBoard

```bash
tensorboard --logdir=tensorboard_logs/
```

Then open: **[http://localhost:6006/](http://localhost:6006/)**

## 🖼️ Example Dashboard

![TensorBoard Scalars](img/tensorboard/tensorboard.png)

*Tip:* Use the **Time Series → Scalars** tab, toggle **Smoothing**, and select multiple runs to compare **Loss/train** curves.

## 🗂️ Project Structure (delta)

```
mlops-gun-detection/
├── tensorboard_logs/         # ← TensorBoard run directories
└── img/
    └── tensorboard/
        └── tensorboard.png   # Screenshot for README
```

## 🚀 Next Stage — API Deployment with FastAPI, Swagger UI & Postman

Next, we’ll build an **inference API** using **FastAPI**, providing a lightweight and production-ready interface for model predictions.  
This stage will include:

* 🧠 Integration of the trained Faster R-CNN model for live inference  
* ⚙️ REST endpoints documented automatically with **Swagger UI**  
* 🧪 Endpoint testing and validation using **Postman**  
* 📦 Container-ready structure for deployment in future stages  

This marks the transition from **model training** to **real-world serving**, enabling external systems to interact with the Gun Detection model via a clean, documented API.
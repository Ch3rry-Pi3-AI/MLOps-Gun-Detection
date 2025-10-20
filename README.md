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

## 🚀 Next Stage — Training Pipeline with DVC

Next, we’ll add a **DVC-powered training pipeline** to version data, params, models, and metrics, and make experiments fully reproducible end-to-end.
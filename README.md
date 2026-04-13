# Chin Detection (YOLO Pose Fine-Tuning)

This project shows the **simplest and fastest way** to fine-tune a **YOLO Pose** model for a chin detection task using keypoints.

---

# 1. Model Preparation

Download pretrained YOLO Pose weights:

```
yolo11s-pose.pt
```

This will be the base model for fine-tuning.

---

# 2. Dataset Structure

Recommended project structure:

```
dataset/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/ 
│
├── labels/
│   ├── train/
│   ├── val/
│
├── data.yaml
├── train_model.ipynb
└── yolo11s-pose.pt
```

---

# 3. Data Annotation

You can label data using CVAT or similar tools.

Each line in a `.txt` file has this format:

```
class x_center y_center width height kpt_x kpt_y visibility
```

### Example:

```
0 0.493021 0.371556 0.05 0.05 0.493021 0.371556 2
```

---

# 4. Annotation Explanation

| Parameter            | Description                 |
| -------------------- | --------------------------- |
| `0`                  | object class (chin)         |
| `x_center, y_center` | center of bounding box      |
| `w, h`               | width and height of the box |
| `kpt_x, kpt_y`       | keypoint coordinates        |
| `visibility`         | keypoint visibility         |
| `0`                  | not labeled                 |
| `1`                  | labeled but not visible     |
| `2`                  | visible                     |

---

# 5. `data.yaml` Configuration

Example:

```yaml
path: dataset

train: images/train
val: images/val
test: images/test

names:
  0: chin

kpt_shape: [1, 3]

flip_idx: [0]
```

---

### Explanation:

* `names` — object classes (only `chin` in this case)
* `kpt_shape: [1, 3]` means:

  * 1 keypoint
  * each keypoint has `(x, y, visibility)`
* `flip_idx` — rule for flipping keypoints horizontally

---

### Example of `flip_idx`

If you had multiple keypoints:

```
nose (0)
left_eye (1)
right_eye (2)
```

Then:

```yaml
flip_idx: [0, 2, 1]
```

---

# 6. Model Training

Install required libraries:

```bash
pip install ultralytics
pip install mlflow
```

Open the notebook:

```
train_model.ipynb
```

Make sure all dataset and model paths are correct before running.

---

# 7. What are Epochs

**Epochs** mean how many times the model goes through the whole dataset.

* small dataset → more epochs
* large dataset → fewer epochs

---

# 8. Training Results

During training, logs are saved in:

```
runs/pose/train/
```

You can find:

* training graphs
* prediction examples (batch images)
* metrics

---

# 9. Saved Model

After training, the best model is saved here:

```
runs/pose/train/weights/best.pt
```

These are the final weights ready for production use.

---

# 10. Model Usage

Example of inference:

```python
from ultralytics import YOLO

model = YOLO("runs/pose/train/weights/best.pt")

results = model("image.jpg")
```

---

# Done

Now you can use the model for chin detection in images and computer vision tasks.

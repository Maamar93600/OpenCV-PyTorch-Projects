# License Plate Detection with YOLO

## 1. Overview

Object detection has many applications in industry. Automatic Number Plate Recognition (ANPR) is used in systems such as Electronic Toll Collection.

License plate detection is an important first step in an ANPR pipeline. The goal of this project is to detect vehicle registration plates in images using deep learning and the Ultralytics YOLO framework.

This project explores different YOLO architectures and input resolutions to study their performance on license plate detection, with particular attention to small objects.

## 2. Objectives

- Train and compare YOLOv8n, YOLO26n, and YOLO26l.
- Evaluate the impact of input resolutions of 320 × 320 and 640 × 640 pixels.
- Use precision, recall, mAP@50, and mAP@50–95 to evaluate detection performance.
- Evaluate the selected model on a separate test dataset.
- Analyze qualitative predictions and identify detection limitations.
- Test the trained model on video inputs.

## 3. Technologies

- Python
- PyTorch
- Ultralytics YOLO
- OpenCV
- TensorBoard
- COCO evaluation metrics

## 4. Dataset

### Training and Validation

The training and validation datasets originate from **Open Images**.

The project focuses on one object class:

- **Class ID:** 0
- **Class name:** Vehicle registration plate

The original bounding-box annotations were converted into the YOLO format. Each bounding box is represented by normalized center coordinates, width, and height.

### Test Dataset

A separate test dataset was obtained from Roboflow Universe:

[License Plate Recognition — Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/12)

The test set was used for the final evaluation and was not used for training.

Since the validation and test datasets originate from different sources, differences in image characteristics, acquisition conditions, and object sizes may affect the comparison of their results.

## 5. Experimental Setup

Five configurations were trained using the Ultralytics YOLO framework.

| Model | Input resolution | Epochs | Batch size |
|---|---:|---:|---:|
| YOLOv8n | 320 × 320 | 30 | 8 |
| YOLOv8n | 640 × 640 | 30 | 8 |
| YOLO26n | 320 × 320 | 30 | 8 |
| YOLO26n | 640 × 640 | 30 | 8 |
| YOLO26l | 640 × 640 | 30 | 6 |

The batch size for YOLO26l was reduced to 6 because of its higher GPU memory requirements. All other configurations used a batch size of 8.

TensorBoard was used to monitor training progress and visualize experiment metrics.

## 6. Results

Detailed quantitative results are presented for **YOLO26l trained at 640 × 640 pixels**. The other configurations were trained for comparison, but their numerical evaluation results are not included here.

### 6.1 Validation Results — Open Images

The validation evaluation included 386 images and 512 annotated instances.

| Metric | Result |
|---|---:|
| Precision | 0.937 |
| Recall | 0.896 |
| mAP@50 | 0.924 |
| mAP@50–95 | 0.695 |

### 6.2 Test Results — Roboflow

The test evaluation included 1,019 images and 1,085 annotated instances.

| Metric | Result |
|---|---:|
| Precision | 0.972 |
| Recall | 0.922 |
| mAP@50 | 0.963 |
| mAP@50–95 | 0.556 |

### 6.3 Detailed COCO Evaluation on the Test Set

| COCO metric | Result |
|---|---:|
| AP@[IoU 0.50:0.95], all objects | 0.555 |
| AP@IoU 0.50, all objects | 0.959 |
| AP@IoU 0.75, all objects | 0.614 |
| AP@[IoU 0.50:0.95], small objects | 0.394 |
| AP@[IoU 0.50:0.95], medium objects | 0.577 |
| AP@[IoU 0.50:0.95], large objects | 0.667 |
| AR@[IoU 0.50:0.95], maxDets=100 | 0.664 |
| AR@IoU 0.50, maxDets=100 | 0.982 |
| AR@IoU 0.75, maxDets=100 | 0.799 |
| AR@[IoU 0.50:0.95], small objects | 0.515 |
| AR@[IoU 0.50:0.95], medium objects | 0.683 |
| AR@[IoU 0.50:0.95], large objects | 0.770 |

**Metric definitions**

- **Precision:** proportion of predicted detections that are correct.
- **Recall:** proportion of ground-truth objects that are detected.
- **mAP@50:** mean Average Precision at an Intersection over Union (IoU) threshold of 0.50.
- **mAP@50–95:** mean Average Precision averaged over IoU thresholds from 0.50 to 0.95.
- **AP:** Average Precision, reported here using COCO evaluation settings.
- **AR:** Average Recall under the specified IoU thresholds and maximum number of detections.

### 6.4 Discussion

The YOLO26l 640 × 640 model achieved a validation mAP@50–95 of 0.695 and a test mAP@50–95 of 0.556.

Precision, recall, and mAP@50 were higher on the Roboflow test set, while mAP@50–95 was lower. Because the validation and test datasets come from different sources, differences in image characteristics, acquisition conditions, and object sizes may contribute to these results.

The COCO metrics also show lower AP for small objects than for medium and large objects. This suggests that small license plates remain more challenging to detect and localize accurately.

Qualitative examples show that some very small license plates were missed. These examples help illustrate the model's limitations, but they represent only a small sample of the test predictions.

## 7. Visual Results

The `Results/` directory contains selected visual outputs from training and evaluation.

- `Results/Train/`: training visualizations and metric curves.
- `Results/Validation/`: selected validation predictions and annotation comparisons.
- `Results/Evaluation/`: evaluation images highlighting predictions and false negatives.
- `Results/Tensorboard/`: a summary plot of training experiments.

## 8. Video Demonstrations

The trained YOLO26l model was also tested on video inputs.

- [YOLO26l license plate detection demo](https://www.youtube.com/watch?v=X3PUhNaNzmo)
- [License plate detection with YOLO26l](https://www.youtube.com/watch?v=FTYfAgQp3Mg)

## 9. Repository Structure

```text
Project-04-Object-Detection/
├── Models/
│   └── best.pt
├── Notebooks/
│   └── Project4_object_detection_YOLO.ipynb
├── Results/
│   ├── Evaluation/
│   ├── Tensorboard/
│   ├── Train/
│   └── Validation/
└── README.md
```

The trained model weights are included in `Models/best.pt`.

The dataset and generated training runs are not included in the repository.

## 10. Future Improvements

- Improve the detection and localization of very small license plates.
- Investigate higher input resolutions and alternative training configurations.
- Analyze false negatives and bounding-box localization errors.
- Evaluate robustness on images acquired under different conditions.

## Author

Maamar

This project is part of my ongoing learning journey in Computer Vision and Deep Learning, with a focus on object detection using PyTorch and Ultralytics YOLO.

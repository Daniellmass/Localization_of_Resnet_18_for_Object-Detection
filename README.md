
# Implementation of Basic Object Detection Model using ResNet18

🎥 [Demo video – Single Object Dectection](https://drive.google.com/file/d/1T6aOgR3NPZOHFWKU_2KDoeoEX4RFyfCX/view)  
🎥 [Demo video – Multi Object Dectection](https://drive.google.com/file/d/1prqulbLfjIk8uoBSdimOPNMfHgkEMz78/view)

## 2.1 Architecture Development

### 2.1.1 Model Building
The MultiBirdModel_Anchors was implemented using a pretrained ResNet-18 as the backbone, following the instruction to build the model using only a pretrained backbone and basic PyTorch components such as Conv2d and ReLU. The classification layers of ResNet-18 were removed, keeping only the convolutional layers up to the last feature map, which outputs a [B, 512, 7, 7] tensor for input images resized to 224×224.

Two task-specific heads were added: one convolutional layer for predicting objectness scores per anchor and grid cell, and another for predicting bounding box parameters (x, y, w, h) for each anchor. These heads were constructed using simple convolutional layers with a kernel size of 3 and padding of 1.

In terms of training strategy, only the early layers of the backbone (layer1 and layer2) were frozen, while deeper layers remained trainable. This decision was based on the idea of reusing general low-level features while allowing the model to adapt higher-level features to the specific task.

The overall design aimed to keep the architecture simple, modular, and consistent with the constraints provided.

### 2.1.2 Axis-Aligned Bounding Boxes (AABB)
In this part of the project, axis-aligned bounding boxes were used to detect a single object category ("Penguin"). The boxes are defined by the top-left corner coordinates (x1, y1) and their dimensions (width, height), forming rectangular regions aligned with the image axes.

The build_targets() function constructs classification and regression targets per grid cell and anchor. For each ground-truth box, its center (cx, cy) and size (w, h) are computed, and the best-fitting anchor is selected based on IoU. The model learns to predict (x1, y1, w, h), from which (x2, y2) can be derived.

The decode_predictions() function applies a sigmoid to the classification scores and filters out boxes with low confidence. The box coordinates are decoded from the predicted tensors and passed to standard non-maximum suppression (NMS), which works with axis-aligned boxes.

### 2.1.3 Loss Functions and Metrics

#### 2.1.3.1 Cross-Entropy Loss
Cross-Entropy Loss for classification was selected due to its suitability in multi-class problems, offering clear probabilistic interpretation and stable gradient computations.

Smooth L1 Loss (Huber Loss) for bounding box regression due to its stability, resistance to outliers, and balanced behavior between small and large prediction errors.

In our task, the goal is to detect a single penguin in each image by predicting both its class (penguin) and its location using a bounding box. Although the problem is simplified to a single object and a single class, the choice of loss functions remains important for achieving stable and accurate results.

**Cross-Entropy Loss for Classification:**
Even though there is only one class (penguin), we still need the model to learn to distinguish between “penguin” and “background” (i.e., no object). Cross-entropy is a standard choice for classification tasks because:

- It provides a clear probabilistic interpretation, encouraging the model to output confidence scores.
- It offers stable and well-behaved gradients, which help with model convergence during training.
- It allows for easy extension to multi-class detection if needed later.

#### 2.1.3.2 Smooth L1 Loss for Bounding Box Regression
When predicting the location of the penguin, we use a bounding box with four values (e.g., x, y, width, height). Smooth L1 loss is ideal for this task because:

- It combines the strengths of L1 and L2 losses, being less sensitive to outliers than L2 and more stable than L1.
- For small errors, it behaves like L2 loss, promoting precision.
- For larger errors, it transitions to L1, reducing the influence of extreme mistakes.
- It is widely used in object detection frameworks (e.g., Fast R-CNN) for its numerical stability and robustness.

Together, these loss functions allow the model to learn both what the object is (classification) and where it is (localization) in a balanced and effective way.

### 2.3.3 Metrics employed
- **Intersection over Union (IoU):** Measures overlap between predicted and ground-truth bounding boxes, crucial for localization accuracy.
- **Mean Average Precision (mAP):** Offers comprehensive evaluation considering both precision and recall, essential for performance assessment in detection task

## 2.4 Data Partitioning
The dataset was strategically split into training (70%), validation (15%), and testing (15%) subsets. This balanced approach ensures that sufficient data is available for training robust models, while maintaining unbiased validation and rigorous testing to accurately evaluate model generalization and prevent overfitting.

## 2.5 Data Augmentation
Albumentations library was chosen explicitly due to its integrated support for handling bounding box transformations automatically, thus preserving annotation integrity during augmentation.

Augmentations implemented include:
- Horizontal Flip
- Gaussian Noise
- Color Jitter (brightness, contrast, saturation, hue)

## 2.6 TensorBoard Monitoring
TensorBoard provided real-time insights into training dynamics, tracking metrics such as loss curves, IoU progression, and mAP scores.

## 2.7 Dataset Selection
The dataset was selected from Roboflow Public Datasets for its relevance to real-world scenarios and appropriate complexity.

## 2.8 Model Training
Training involved the Adam optimizer, a learning rate of 1e-4, batch size of 30, and a step learning rate scheduler (StepLR), decreasing learning rate periodically for finer parameter tuning.

## 2.9 Inference on External Video
The model's performance was assessed on an external video different from the training dataset to determine real-world applicability.

- Hyperlink – part 2
- Code for - part 2 (Google Collab)

# 3 Transitioning from Single-Object to Multi-Object Detection using ResNet18

## 3.1 Overview of Significant Changes Implemented After Part 2
We transitioned to the Pascal VOC format dataset, which provides richer annotations for multi-object detection. It uses XML files specifying bounding boxes in [xmin, ymin, xmax, ymax] format.

## 3.2 Data Augmentation
Enhancements include:
- Resize to 224x224
- Horizontal Flip
- Gaussian Noise
- Color Jitter

## 3.3 Anchors Implementation
Introduced anchor sizes: 32x32, 64x64, 96x96.

## 3.4 Model Architecture Adjustments
Added convolutional layers for predicting classification scores and bounding box coordinates for each anchor.

## 3.4 Loss Functions Modification
- **BCEWithLogitsLoss** for classification
- **Smooth L1 Loss** for regression

## 3.5 Target Generation (Build Targets)
Best anchor selected per ground truth box using IoU.

## 3.6 Decoding Predictions and Non-Maximum Suppression (NMS)
Used to eliminate overlapping boxes and retain high-confidence predictions.

## 3.7 Metrics and Validation
Used **mAP@0.5**. Validation scores were 0.5–0.6, but real-life performance was satisfactory.

- Hyperlink – part 3
- Code for - part 3 (Google Collab)

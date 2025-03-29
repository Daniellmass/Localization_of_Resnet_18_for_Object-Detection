# -*- coding: utf-8 -*-
"""Another copy of ObjDetection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fSkA2DWEdD28_B9mPEVywFSr-4D2kLjB
"""

from google.colab import drive
drive.mount('/content/drive')

!ls /content/drive/MyDrive/merged_images/db2

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import os
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import albumentations
print(albumentations.__version__)

!ls /content/drive/MyDrive/merged_images/merged_train.json

#@title testing functions

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def run_visualize_test(model_path, test_images_dir, test_json_path, device='cpu'):
    # 1) Build or load your model, same arch as training
    model = PenguinDetectionModel(num_classes=2)  # or whatever your class is named
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2) Transforms for test: typically just resize+normalize
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    # 3) Create dataset
    test_dataset = COCO_TestPenguinDataset(
        images_dir=test_images_dir,
        json_file=test_json_path,
        transform=test_transform
    )

    print("Filtered test dataset length:", len(test_dataset))

    # 4) Pick a few samples to visualize
    indices_to_show = list(range(20))
    # indices_to_show = [0, 1, 2, 3,5,6,7,8]  # or random subset
    for idx in indices_to_show:
        image_tensor, label, gt_bbox, filename = test_dataset[idx]
        print(f"\n=== Sample {idx} / File: {filename} ===")
        # 5) Visualize
        visualize_single_prediction(model, image_tensor, gt_bbox, test_transform, device)


def visualize_single_prediction(model, image_tensor, gt_bbox, transform, device):
    model.eval()

    # 1) Un-normalize the image tensor for display
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    denorm_image = inv_normalize(image_tensor).clamp(0,1)
    np_image = denorm_image.permute(1,2,0).cpu().numpy()

    # 2) Convert for inference (the model expects a normalized tensor)
    # But your 'image_tensor' is already normalized & resized, so we can just pass it as is.
    image_batch = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        cls_logits, bbox_pred = model(image_batch)

    pred_bbox = bbox_pred[0].cpu().numpy()
    pred_class = torch.argmax(cls_logits, dim=1)[0].item()

    # 3) Plot
    fig, ax = plt.subplots(1, figsize=(6,6))
    ax.imshow(np_image)

    # Draw the GT bbox in red
    gx, gy, gw, gh = gt_bbox.numpy()
    rect_gt = patches.Rectangle((gx, gy), gw, gh, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect_gt)

    # Draw the predicted box in blue
    px, py, pw, ph = pred_bbox
    rect_pred = patches.Rectangle((px, py), pw, ph, linewidth=2, edgecolor='b', facecolor='none')
    ax.add_patch(rect_pred)

    # Title shows predicted class
    label_text = "Penguin" if pred_class==1 else f"Class:{pred_class}"
    ax.set_title(f"Pred: {label_text}\nGT: {gx:.1f},{gy:.1f},{gw:.1f},{gh:.1f}\nPred: {px:.1f},{py:.1f},{pw:.1f},{ph:.1f}")
    plt.show()

#@title debugStuff


def visualize_original_annotation(images_dir, json_file, image_id):
    """
    Loads an image and its ground truth bounding box (without any transforms)
    and plots them together.

    Parameters:
      images_dir (str): Directory where the images are stored.
      json_file (str): Path to the COCO-format JSON file.
      image_id (int): The ID of the image to visualize.
    """
    # Load the JSON annotations.
    with open(json_file, 'r') as f:
        coco = json.load(f)

    # Find the image info with the specified image_id.
    image_info = next((img for img in coco['images'] if img['id'] == image_id), None)
    if image_info is None:
        print(f"Image id {image_id} not found.")
        return

    # Construct the full image path and load the image.
    image_path = os.path.join(images_dir, image_info['file_name'])
    image = Image.open(image_path).convert("RGB")

    # Retrieve the annotations for this image.
    anns = [ann for ann in coco['annotations'] if ann['image_id'] == image_id]
    if not anns:
        print(f"No annotations found for image id {image_id}.")
        return

    # For this example, assume we are interested in the first annotation.
    ann = anns[0]
    bbox = ann['bbox']  # Typically in [x, y, width, height] format.

    # Plot the image and draw the bounding box.
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)
    x, y, w, h = bbox
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"Image ID {image_id} - Ground Truth")
    plt.show()

# Example usage:
# Change the paths and image_id to one that exists in your dataset.
# visualize_original_annotation("db1/train", "db1/train/_annotations.coco.json", image_id=353)


def run_inference(image_path, model, transform):
    """
    Loads a single image, applies the transform, and runs inference on it.
    Returns the predicted bounding box and predicted class.
    """
    # Load the image using PIL
    image = Image.open(image_path).convert("RGB")

    # Save a copy of the original image for visualization.
    orig_image = image.copy()

    # Apply the same transform used during validation (without data augmentation)
    image_tensor = transform(image).unsqueeze(0)  # add batch dimension

    # Run inference (with no gradient computation)
    model.eval()  # ensure model is in eval mode
    with torch.no_grad():
        cls_logits, bbox_pred = model(image_tensor)

    # Get the predicted bounding box (assumes bbox_pred shape is [1, 4])
    bbox = bbox_pred[0].cpu().numpy()

    # Get predicted class label (0=background, 1=penguin)
    pred_class = torch.argmax(cls_logits, dim=1)[0].item()

    return orig_image, bbox, pred_class

def visualize_prediction(image, bbox, pred_class):
    """
    Displays the image with the predicted bounding box.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Here we assume bbox is in [x, y, width, height] format.
    x, y, w, h = bbox
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    label_text = "penguin" if pred_class == 1 else "background"
    plt.title(f"Predicted: {label_text}")
    plt.show()

def visualize_tensor(image_tensor):
    # image_tensor: shape [3, H, W], normalized
    # Build an inverse transform with the negative of the mean/std used
    inv_normalize = T.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    )

    # Apply inverse normalization
    inv_tensor = inv_normalize(image_tensor.clone())

    # Clamp to [0,1] range just in case
    inv_tensor = inv_tensor.clamp(0, 1)

    # Convert to numpy for matplotlib
    np_img = inv_tensor.permute(1, 2, 0).cpu().numpy()
    return np_img



def visualize_ground_truth(dataset, idx):
    image, label, bbox = dataset[idx]
    image = visualize_tensor(image)
    # Undo the normalization if necessary (this example assumes no normalization for simplicity)
    plt.figure(figsize=(6,6))
    plt.imshow(transforms.ToPILImage()(image))
    ax = plt.gca()
    x, y, w, h = bbox.numpy()
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.title("Ground Truth")
    plt.show()

def visualize_image_with_annotation(image, gt_bbox, pred_bbox=None, pred_class=None):
    """
    Displays an image with the ground truth bounding box.
    If provided, also overlays the predicted bounding box and class.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Ground truth bbox in red
    gt_x, gt_y, gt_w, gt_h = gt_bbox
    gt_rect = patches.Rectangle((gt_x, gt_y), gt_w, gt_h, linewidth=2, edgecolor='r', facecolor='none', label='Ground Truth')
    ax.add_patch(gt_rect)

    if pred_bbox is not None:
        pred_x, pred_y, pred_w, pred_h = pred_bbox
        pred_rect = patches.Rectangle((pred_x, pred_y), pred_w, pred_h, linewidth=2, edgecolor='b', facecolor='none', label='Prediction')
        ax.add_patch(pred_rect)
        label_text = "penguin" if pred_class == 1 else "background"
        ax.set_title(f"Predicted: {label_text}")

    plt.legend()
    plt.show()

def debug_on_sample(model, dataset, transform, sample_index=0):
    """
    Visualizes a sample image with both its ground truth and predicted bounding boxes.
    """
    image, label, gt_bbox = dataset[sample_index]

    # Convert tensor image back to PIL image for visualization
    # (Assuming the image was normalized; you may need to reverse the normalization.)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    image_denorm = inv_normalize(image).clamp(0, 1)
    image_pil = transforms.ToPILImage()(image_denorm)

    # Run inference on the same image
    image_tensor = transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        cls_logits, bbox_pred = model(image_tensor)
    pred_bbox = bbox_pred[0].cpu().numpy()
    pred_class = torch.argmax(cls_logits, dim=1)[0].item()

    visualize_image_with_annotation(image_pil, gt_bbox.numpy(), pred_bbox, pred_class)


def print_sample_predictions(model, dataset, transform, sample_indices=[0, 1, 2, 3, 4]):
    """
    For each sample index in sample_indices, prints the ground truth and predicted bounding boxes and class,
    then visualizes the image with both the ground truth (red) and predicted (blue) bounding boxes.
    """
    model.eval()
    for idx in sample_indices:
        # Retrieve the sample (already transformed and resized) from the dataset.
        image_tensor, label, gt_bbox = dataset[idx]
        print(f"\nSample {idx}:")
        print("Ground Truth bbox:", gt_bbox)

        # Convert tensor image back to PIL image for visualization.
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        image_denorm = inv_normalize(image_tensor).clamp(0, 1)
        image_pil = transforms.ToPILImage()(image_denorm)

        # Run inference on the image.
        image_input = transform(image_pil).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            cls_logits, bbox_pred = model(image_input)
        pred_bbox = bbox_pred[0].cpu().numpy()
        pred_class = torch.argmax(cls_logits, dim=1)[0].item()

        print("Predicted bbox:", pred_bbox)
        print("Predicted class:", pred_class)

        # Visualize the results.
        visualize_image_with_annotation(image_pil, gt_bbox.numpy(), pred_bbox, pred_class)

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms

def debug_bbox(dataset, index):
    """
    Fetch the transformed image & scaled bbox from dataset[index],
    undo the normalization, and plot the bounding box.
    """
    image_tensor, label, bbox = dataset[index]

    # Inverse normalization: (assuming your mean=[0.485, 0.456, 0.406],
    # std=[0.229, 0.224, 0.225]). Adjust if yours differ.
    inv_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    # Undo normalization & clamp
    image_denorm = inv_norm(image_tensor).clamp(0, 1)
    # Convert tensor -> HWC -> NumPy
    image_np = image_denorm.permute(1, 2, 0).cpu().numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_np)

    x, y, w, h = bbox.numpy()
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    ax.set_title(f"Index {index}, label={label.item()}")
    plt.show()

#@title PenguinDataset_old

class COCO_PenguinDataset(Dataset):
    def __init__(self, images_dir, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.coco = json.load(f)
        self.images_dir = images_dir
        self.transform = transform

        # Create a mapping from image_id to a list of its annotations.
        ann_map = {}
        for ann in self.coco['annotations']:
            ann_map.setdefault(ann['image_id'], []).append(ann)

        # Filter images:
        # 1. The image must have annotations.
        # 2. It must have exactly one annotation.
        # 3. That annotation's category_id must be 1 (penguin).
        self.images = [img for img in self.coco['images']
                       if img['id'] in ann_map
                       and len(ann_map[img['id']]) == 1
                       and ann_map[img['id']][0]['category_id'] == 1]

        # For convenience, store the single annotation per image.
        self.annotations = {img['id']: ann_map[img['id']][0] for img in self.images}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
      img_info = self.images[idx]
      img_path = os.path.join(self.images_dir, img_info['file_name'].replace('\\', '/'))

      # Load original image
      image = Image.open(img_path).convert("RGB")
      orig_width, orig_height = image.size

      # Get the single annotation
      ann = self.annotations[img_info['id']]
      bbox_orig = ann['bbox']  # [x, y, w, h]

      # We want our final images to be 224×224 via the transform,
      # so we’ll compute how much we *would* scale the original box
      # to fit the 224×224 image:
      target_width, target_height = 224.0, 224.0
      scale_x = target_width / orig_width
      scale_y = target_height / orig_height

      x, y, w, h = bbox_orig
      x_scaled = x * scale_x
      y_scaled = y * scale_y
      w_scaled = w * scale_x
      h_scaled = h * scale_y

      # Now, DO NOT manually resize here. Just let the transform do it:
      # image = image.resize((224, 224), Image.BILINEAR)   # <-- REMOVE THIS

      # Use your transform pipeline (which includes transforms.Resize((224,224))):
      if self.transform:
          image = self.transform(image)  # This will do the 224 resize

      bbox = torch.tensor([x_scaled, y_scaled, w_scaled, h_scaled], dtype=torch.float32)
      label = torch.tensor(1, dtype=torch.long)  # Single-class: 1 = penguin

      return image, label, bbox

#@title PenguinDataset

import torch
import json
import cv2
import numpy as np
import os

class COCO_PenguinDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.coco = json.load(f)

        self.images_dir = images_dir
        self.transform = transform

        # Build annotation map
        ann_map = {}
        for ann in self.coco['annotations']:
            ann_map.setdefault(ann['image_id'], []).append(ann)

        # Filter images to those that have exactly 1 annotation with category_id=1, etc.
        self.images = []
        self.annotations = {}
        for img in self.coco['images']:
            img_id = img['id']
            # If the image has any annotations at all:
            if img_id in ann_map:
                # Check how many are category_id=1 (penguin)
                penguin_anns = [ann for ann in ann_map[img_id] if ann['category_id'] == 1]

                if len(penguin_anns) == 1:
                    # Exactly one penguin => keep as positive example
                    self.images.append(img)
                    self.annotations[img_id] = penguin_anns[0]  # store that annotation
                elif len(penguin_anns) == 0:
                    # No penguins or multiple penguins => treat as negative or skip
                    # If you want negative examples, just store None
                    self.images.append(img)
                    self.annotations[img_id] = None
            else:
                # No annotations => definitely negative example
                self.images.append(img)
                self.annotations[img_id] = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']

        # Build the proper path & load image with OpenCV
        fname = img_info['file_name'].replace('\\', '/')
        img_path = os.path.join(self.images_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not open image at {img_path}")
            # Optionally skip or try next sample:
            return self.__getitem__((idx + 1) % len(self))

        # Convert BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Retrieve annotation (None => negative sample)
        ann = self.annotations[img_id]

        if ann is None:
            # No penguin => negative => empty bounding boxes
            bboxes = []
            class_labels = []
        else:
            # We have exactly one penguin annotation => clamp it
            x, y, bw, bh = ann['bbox']
            x2 = x + bw
            y2 = y + bh
            x = max(0, x)
            y = max(0, y)
            x2 = min(x2, w)
            y2 = min(y2, h)

            bw = x2 - x
            bh = y2 - y

            if bw <= 1 or bh <= 1:
                print(f"img id: {img_id}  Degenerate box after clipping: x={x}, y={y}, w={bw}, h={bh}")
                # Try another sample or handle differently
                return self.__getitem__((idx + 1) % len(self))

            bboxes = [[x, y, bw, bh]]
            class_labels = [1]

        # Albumentations transform
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            image_out = transformed['image']
            out_bboxes = transformed['bboxes']
            out_labels = transformed['class_labels']
        else:
            # No transform => just turn image into a torch tensor
            image_out = torch.from_numpy(image).permute(2,0,1).float()
            out_bboxes = bboxes
            out_labels = class_labels

        # If no boxes remain, set label=0 + [0,0,0,0]
        if len(out_bboxes) == 0:
            final_label = 0
            final_box = [0,0,0,0]
        else:
            # We have exactly 1 box
            x_new, y_new, w_new, h_new = out_bboxes[0]
            final_label = out_labels[0]
            final_box = [x_new, y_new, w_new, h_new]

        # Convert to torch Tensors
        label_tensor = torch.tensor(final_label, dtype=torch.long)
        bbox_tensor = torch.tensor(final_box, dtype=torch.float32)

        return image_out, label_tensor, bbox_tensor

#@title TestPenguinDataset


import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class COCO_TestPenguinDataset(Dataset):
    def __init__(self, images_dir, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.coco = json.load(f)

        self.images_dir = images_dir
        self.transform = transform

        # Create a mapping from image_id to a list of its annotations
        ann_map = {}
        for ann in self.coco['annotations']:
            ann_map.setdefault(ann['image_id'], []).append(ann)

        # Filter images that have exactly 1 annotation that is category_id=1, etc.
        # (If you truly have multiple bounding boxes per image, you can adapt logic below.)
        self.images = []
        self.annotations = {}
        for img in self.coco['images']:
            img_id = img['id']
            if img_id in ann_map and len(ann_map[img_id]) == 1:
                # Possibly also check ann_map[img_id][0]['category_id'] == 1
              self.images.append(img)
              self.annotations[img_id] = ann_map[img_id][0]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        # Load the original image
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        ann = self.annotations[img_id]
        # [x, y, w, h] in original resolution
        bbox_orig = ann['bbox']

        # We'll let our transform do the resizing to (224,224).
        # Compute how we scale the bounding box from orig→224 if you want to feed
        # directly scaled bboxes to your loss or debug. But for visualization, you can do
        # something simpler: store the raw box, and do the same approach as training if
        # you need consistency.

        # For demonstration, let's do the same approach as your training set:
        scale_x = 224.0 / orig_w
        scale_y = 224.0 / orig_h
        x, y, w, h = bbox_orig
        x_scaled = x * scale_x
        y_scaled = y * scale_y
        w_scaled = w * scale_x
        h_scaled = h * scale_y

        if self.transform:
            image = self.transform(image)

        # Return the scaled box to match the transformed image shape (224×224).
        bbox = torch.tensor([x_scaled, y_scaled, w_scaled, h_scaled], dtype=torch.float32)
        label = torch.tensor(1, dtype=torch.long)  # single class

        return image, label, bbox, img_info['file_name']  # also return filename if you like

#@title PenguinDetectionModel

# --- Model definition (same as before) ---
class PenguinDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(PenguinDetectionModel, self).__init__()
        backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        # Freeze layers
        for param in self.backbone[0].parameters():
            param.requires_grad = False

        # Now we have feature maps of shape [batch, 512, 7, 7] for input 224x224
        self.conv_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # or keep a 3x3 or so
        )
        self.fc_cls = nn.Linear(256, num_classes)
        self.fc_bbox = nn.Linear(256, 4)


    def forward(self, x):
        # 1) Pass through the ResNet backbone (up to the last conv block).
        features = self.backbone(x)  # shape [B, 512, 7, 7] for 224×224 input

        # 2) Pass through the new conv head (conv + ReLU + AdaptiveAvgPool2d).
        out = self.conv_head(features)  # shape [B, 256, 1, 1]

        # 3) Flatten from [B, 256, 1, 1] to [B, 256].
        out = out.view(out.size(0), -1) # shape [B, 256]

        # 4) Class logits & bounding box predictions.
        cls_logits = self.fc_cls(out)   # shape [B, num_classes]
        bbox_pred = self.fc_bbox(out)   # shape [B, 4]
        return cls_logits, bbox_pred

#@title losses and train functions

# --- Loss & metrics (same as before) ---
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_br, y1_br = x1 + w1, y1 + h1
    x2_br, y2_br = x2 + w2, y2 + h2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_br, x2_br)
    inter_y2 = min(y1_br, y2_br)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def calculate_map(pred_boxes, gt_boxes, iou_threshold=0.5):
    ious = [compute_iou(p, g) for p, g in zip(pred_boxes, gt_boxes)]
    return sum(ious) / len(ious) if ious else 0

# --- Training and validation functions ---
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    cls_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()

    for images, labels, bboxes in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)

        optimizer.zero_grad()
        cls_logits, bbox_preds = model(images)
        loss_cls = cls_loss_fn(cls_logits, labels)
        loss_bbox = bbox_loss_fn(bbox_preds, bboxes)
        loss_cls_weight = 1.0
        loss_bbox_weight = 5.0  # or something you tune
        loss = loss_cls_weight * loss_cls + loss_bbox_weight * loss_bbox
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_iou = 0.0
    pred_boxes = []
    gt_boxes = []

    total_coord_diff = 0.0
    count = 0

    with torch.no_grad():
        for images, labels, bboxes in dataloader:
            images = images.to(device)
            bboxes = bboxes.to(device)

            cls_logits, bbox_preds = model(images)

            for pred, gt in zip(bbox_preds, bboxes):
                # Compute IoU
                pred_np = pred.cpu().numpy()
                gt_np   = gt.cpu().numpy()
                total_iou += compute_iou(pred_np, gt_np)

                # Collect for mAP
                pred_boxes.append(pred_np)
                gt_boxes.append(gt_np)

                # -- Coordinate error (mean absolute difference)
                diff = (pred - gt).abs().mean().item()
                total_coord_diff += diff
                count += 1

    # Average IoU and mAP
    avg_iou = total_iou / len(dataloader.dataset)
    mAP = calculate_map(pred_boxes, gt_boxes)

    # Mean coordinate error
    avg_coord_error = total_coord_diff / count if count > 0 else 0.0

    return avg_iou, mAP, avg_coord_error


def train_model(device,train_loader,val_loader):
    writer = SummaryWriter(log_dir="/content/drive/MyDrive/runs/penguin_detection")

    model = PenguinDetectionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    num_epochs = 50

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        avg_iou, mAP, avg_coord_error = validate(model, val_loader, device)

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']



        # Log metrics to TensorBoard.
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Metrics/IoU", avg_iou, epoch)
        writer.add_scalar("Metrics/mAP", mAP, epoch)
        writer.add_scalar("Metrics/BoxCoordError", avg_coord_error, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {train_loss:.4f} |  LR: {current_lr} "
              f"IoU: {avg_iou:.4f} | mAP: {mAP:.4f} | CoordErr: {avg_coord_error:.4f}")
    # Save the model.
    torch.save(model.state_dict(), "/content/drive/MyDrive/penguin_detection_model_0310_with_false.pth")
    print("Training complete and model saved.")
    writer.close()

#@title video

import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms

def inference_on_video(
    model,
    video_path,
    output_video_path,
    device='cpu',
    confidence_threshold=0.5
):
    """
    Runs the given model on each frame of a video. Draws bounding boxes
    for frames where predicted class=1 or predicted confidence> threshold,
    and writes out an annotated video.

    Args:
      model (nn.Module): Trained detection model, in eval mode
      video_path (str): Path to input video file
      output_video_path (str): Path where annotated video will be saved
      device (str): "cpu" or "cuda"
      confidence_threshold (float): If you had a separate confidence or
        softmax threshold. (If your model is just binary 'class=1 vs 0=bg',
        you can rely on argmax or a logit difference.)
    """

    # 1) Prepare transforms: same as your test/val
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    # 2) Setup video capture & writer
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or "XVID" etc.
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (frame_w, frame_h)
    )

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3) Convert current frame to the format your model expects
        #    (PIL -> transform -> Tensor)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        image_tensor = transform(pil_image).unsqueeze(0).to(device)  # shape=[1,3,224,224]

        # 4) Inference
        with torch.no_grad():
            cls_logits, bbox_pred = model(image_tensor)

        # predicted class
        pred_class = torch.argmax(cls_logits, dim=1).item()  # 0 or 1
        # if you have a predicted probability or "confidence" in that class:
        probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]
        penguin_prob = probs[1]  # prob the class is 1

        # bounding box
        bbox = bbox_pred[0].cpu().numpy()  # [x, y, w, h] in 224×224 coords

        # 5) If predicted class=1 and penguin_prob> confidence_threshold:
        if pred_class == 1 and penguin_prob > confidence_threshold:
            # We'll need to rescale the bbox from 224×224 -> original frame size
            # The frame is (frame_w, frame_h).
            scale_x = frame_w / 224
            scale_y = frame_h / 224

            x, y, w, h = bbox
            x1 = int(x * scale_x)
            y1 = int(y * scale_y)
            x2 = int((x + w) * scale_x)
            y2 = int((y + h) * scale_y)

            # 6) Draw bounding box in BGR color
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255), 2  # red rectangle
            )
            # Optionally put text
            label_text = f"Penguin: {penguin_prob:.2f}"
            cv2.putText(
                frame,
                label_text,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # 7) Write annotated frame to output video
        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    print(f"Inference complete! Annotated video saved as: {output_video_path}")

#@title main

import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Main pipeline with TensorBoard logging ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup TensorBoard writer.

    # Modify these paths to point to your actual directories and JSON files.
    train_images_dir = "/content/drive/MyDrive/merged_images"
    train_json = "/content/drive/MyDrive/merged_images/merged_train.json"
    val_images_dir = "/content/drive/MyDrive/merged_images_val"  # Assuming you have a validation split.
    val_json = "/content/drive/MyDrive/merged_images_val/merged_train_valid.json"

    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), mean=0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
        bbox_params=A.BboxParams(
            format='coco',            # [x_min, y_min, width, height]
            label_fields=['class_labels'],
            min_area=0.0,
            min_visibility=0.0,
        )
    )


    val_transform = A.Compose([
                                A.Resize(224,224),
                                A.Normalize(mean=(0.485,0.456,0.406),
                                            std=(0.229,0.224,0.225)),
                                ToTensorV2()
                                  ],
                                      bbox_params=A.BboxParams(
                                          format='coco',
                                          label_fields=['class_labels'],
                                          min_area=0.0,
                                          min_visibility=0.0,
                                      )
                                  )


    # Create datasets.
    print(train_json)
    print(train_images_dir)

    train_dataset = COCO_PenguinDataset(
                                            images_dir=train_images_dir,
                                            json_file=train_json,
                                            transform=train_transform
                                        )
    val_dataset = COCO_PenguinDataset(
                                            images_dir=val_images_dir,
                                            json_file=val_json,
                                            transform=val_transform
                                        )

    print(len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=30, shuffle=False, num_workers=4)

    # Initialize the model.
    train_model(device,train_loader,val_loader)

        # Make sure the model architecture matches the one you trained.
    model = PenguinDetectionModel()
    model.load_state_dict(torch.load("/content/drive/MyDrive/penguin_detection_model_0310_with_false.pth", map_location=torch.device('cpu')))
    model.to(device)
    model.eval()  # set to evaluation mode

    inference_on_video(
        model=model,
        video_path="/content/drive/MyDrive/Emperor.mp4",
        output_video_path="/content/drive/MyDrive/single2_peng_out.mp4",
        device=device,
        confidence_threshold=0.5
    )


    # run_visualize_test(
    # model_path="/content/drive/MyDrive/penguin_detection_model_0310_with_false.pth",
    # test_images_dir="/content/drive/MyDrive/penguin_images_test",
    # test_json_path="/content/drive/MyDrive/penguin_images_test/_annotations_test.json",
    # device="cuda"  # or "cpu")
    # )
if __name__ == "__main__":
    main()
# Deepfake Image & Video Analyzer
A research-oriented deepfake detection pipeline for classifying images as **pristine (real)** or **fake**, built around the methodology and implementation of the **DFAD 2023 — DeepFake Analysis and Detection** challenge.

The repository contains the preprocessing pipeline, augmentation strategies, model architectures, training code, validation-set construction, inference/evaluation scripts, ensemble prediction support, error analysis utilities, and the associated research paper.

> **Research attribution:** The core implementation is based on the DFAD 2023 work by Davide Alessandro Coccomini, Giuseppe Amato, Fabrizio Falchi, and Claudio Gennaro from ISTI-CNR, which achieved **1st place at DFAD 2023**. This repository should therefore be understood as a packaged research implementation and experimentation repository, not as an independently reproduced benchmark.

---

# 1. Project Overview
Deepfakes are synthetic or manipulated images that can look visually convincing while containing subtle artifacts introduced during generation or manipulation.

The central objective of this project is to train a computer-vision model to distinguish between:

```
PRISTINE IMAGE
      vs
FAKE / SYNTHETIC IMAGE
```
Instead of relying on a single visual cue, the training pipeline exposes the model to a wide range of image transformations so that it can learn more robust representations.

The overall system can be summarized as:

```
                 INPUT IMAGE
                      │
                      ▼
              Image Preprocessing
                      │
                      ▼
             Data Augmentation
                      │
                      ▼
            Vision Classification
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
    Cross-Efficient  ResNet-50    Swin
         ViT                       Transformer
          │           │           │
          └───────────┼───────────┘
                      ▼
                 Model Logit
                      │
                      ▼
                Sigmoid Score
                      │
                      ▼
                 Threshold
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
         PRISTINE              FAKE
```

---

# 2. Main Idea
A deepfake detector should not simply memorize the visual appearance of the training dataset.

For example, if all fake images in training have a particular compression pattern, the model may learn that compression artifact rather than learning useful forensic characteristics.

This project therefore uses **aggressive data augmentation and multiple model architectures** to improve robustness and investigate generalization.

The training pipeline introduces transformations such as:

- Image compression
- Gaussian noise
- ISO noise
- Multiplicative noise
- Horizontal flipping
- Resizing
- Random cropping
- Brightness and contrast changes
- Color transformations
- Blur
- Cutout / coarse dropout
- Grayscale
- Sepia
- Shadows
- Gamma changes
- Rotation and scaling
- FFT
- Optional DCT transformation
The resulting system is designed around the question:

> **Can the detector distinguish fake imagery beyond the exact visual characteristics it saw during training?**

---

# 3. Complete System Workflow

## Step 1 — Input Dataset
The model operates on labeled image samples.

The dataset is represented through CSV files containing image paths and their corresponding labels.

Conceptually:

```
image_path                         label
-----------------------------------------
dataset/real/img001.jpg              0
dataset/real/img002.jpg              0
dataset/fake/img001.jpg              1
dataset/fake/img002.jpg              1
```
The project uses:

```
0 → Pristine / Real
1 → Fake
```
The training script loads the CSV using Pandas and creates a `DeepFakesDataset`.

---

# 4. Dataset Loading Workflow
The dataset class supports two loading strategies.

### Normal loading
Images remain on disk and are read when the DataLoader requests them.

```
CSV
 │
 ▼
Image Path
 │
 ▼
OpenCV reads image
 │
 ▼
Preprocessing
 │
 ▼
Model
```
This reduces memory usage and is suitable for larger datasets.

### Pre-loaded mode
Images can instead be loaded into memory before training.

```
CSV
 │
 ▼
Read all images
 │
 ▼
RAM
 │
 ▼
DataLoader
 │
 ▼
Preprocessing
 │
 ▼
Model
```
This can reduce repeated disk I/O but requires substantially more memory.

The behaviour is controlled through:

```
--pre_load_images
```

---

# 5. Image Preprocessing
Before an image is passed to the vision model, it is converted into the spatial representation expected by the network.

The main configuration uses:

```
Image Size = 224 × 224
```
The general preprocessing workflow is:

```
Original Image
      │
      ▼
Isotropic Resize
      │
      ▼
Preserve Aspect Ratio
      │
      ▼
Padding if Necessary
      │
      ▼
224 × 224 Image
```
The project uses an `IsotropicResize` operation followed by padding rather than simply stretching every image directly into a square.

This reduces unnecessary geometric distortion.

---

# 6. Training Augmentation Workflow
Training augmentation is one of the most important parts of the project.

Instead of showing the model the same image representation every time, the training pipeline can generate different variations.

```
                     Original Image
                           │
           ┌───────────────┼────────────────┐
           │               │                │
           ▼               ▼                ▼
      Compression        Noise          Geometric
                                        Transformations
           │               │                │
           ├───────────────┼────────────────┤
                           ▼
                    Color / Lighting
                           │
                           ▼
                     Blur / Occlusion
                           │
                           ▼
                 Frequency Transformation
                           │
                           ▼
                    Augmented Image
```

### Examples of transformations

```
Compression
Noise
Resize
Crop
Flip
Brightness
Contrast
Hue/Saturation
Blur
Cutout
Grayscale
Sepia
Shadow
Gamma
Rotation
Scaling
FFT
DCT
```
The implementation in `deepfakes_dataset.py` uses Albumentations to construct these transformation pipelines.

---

# 7. Image Modes
The dataset implementation provides different processing modes.

## Image Mode 0
The primary augmentation pipeline is used.

It includes transformations such as:

```
Compression
↓
Noise
↓
Flip
↓
Resize / Crop
↓
Brightness / Contrast / Color
↓
Blur
↓
Dropout
↓
Grayscale / Sepia
↓
Shadow / Gamma
↓
Rotation / Scaling
↓
FFT
```
This is intended to expose the model to a wide range of visual variations.

## Image Mode 1
This mode adds DCT processing.

```
Spatial Image
      +
DCT Representation
```
The implementation applies a DCT transform during preprocessing.

This is useful because image-generation and manipulation processes can introduce frequency-domain artifacts that are not always obvious in the raw RGB image.

---

# 8. Why Frequency-Domain Processing?
Traditional image inspection mainly looks at pixel-space information.

However, synthetic imagery can contain subtle high-frequency or compression-related patterns that are easier to analyze in transformed domains.

The project therefore contains:

```
Spatial Domain
       +
Frequency Domain
       │
       ├── FFT
       └── DCT
```
FFT and DCT processing are incorporated into the augmentation / preprocessing pipeline rather than being treated as a separate standalone detector.

---

# 9. Model Architecture
The repository supports three model choices.

```
                 MODEL SELECTION
                       │
        ┌──────────────┼───────────────┐
        │              │               │
        ▼              ▼               ▼
Cross-Efficient      ResNet-50       Swin Transformer
     ViT
        │              │               │
        └──────────────┼───────────────┘
                       ▼
                 Binary Output
                       │
                       ▼
                   Fake Score
```
The model is selected through:

```
--model
```
with:

```
0 → Cross-Efficient ViT
1 → ResNet-50
2 → Swin Transformer
```

---

# 10. Cross-Efficient ViT
The repository includes a dedicated implementation under:

```
cross-efficient-vit/
```
The Cross-Efficient ViT architecture combines convolutional feature extraction with transformer-based processing.

The configuration exposes parameters for:

```
High-resolution representation
Low-resolution representation
Patch size
Encoder depth
Attention heads
MLP dimensions
Cross-attention
Dropout
```
The purpose of the multi-scale design is to process information at different resolutions and allow interaction between feature representations.

---

# 11. ResNet-50
The ResNet-50 option uses an ImageNet-pretrained ResNet-50 backbone.

The original classification layer is replaced by a single-output layer:

```
ResNet-50
    ↓
2048-dimensional feature representation
    ↓
Linear Layer
    ↓
1 output logit
```
That single logit represents the model's binary classification output.

---

# 12. Swin Transformer
The repository also supports:

```
swin_base_patch4_window7_224.ms_in22k_ft_in1k
```
through `timm`.

The classification head is replaced with a single output.

Conceptually:

```
Input Image
     ↓
Patch Partition
     ↓
Swin Transformer Blocks
     ↓
Hierarchical Feature Representation
     ↓
Classification Head
     ↓
1 Logit
```
The included research paper reports that Swin Transformer was the strongest architecture among the architectures explored in the original DFAD 2023 work.

That statement refers to the **original research**, not to a newly reproduced benchmark in this repository.

---

# 13. Training Pipeline
The main training entry point is:

```
train.py
```
The complete training workflow is:

```
                  Training CSV
                       │
                       ▼
               Pandas DataFrame
                       │
                       ▼
              Shuffle Training Set
                       │
                       ▼
              DeepFakesDataset
                       │
                       ▼
              Data Augmentation
                       │
                       ▼
                   DataLoader
                       │
                       ▼
                 Vision Model
                       │
                       ▼
                  Logit Output
                       │
                       ▼
             BCEWithLogitsLoss
                       │
                       ▼
                 Backpropagation
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
         Optimizer           LR Scheduler
             │                   │
             └─────────┬─────────┘
                       ▼
                 Next Training Step
                       │
                       ▼
                  Validation
                       │
                       ▼
           Validation Loss / F1
                       │
                       ▼
            Improved Validation?
                  /        \
                YES         NO
                 │           │
                 ▼           ▼
            Save Model    Track Patience
                             │
                             ▼
                       Early Stopping
```

---

# 14. Loss Function
The implementation uses:

```
torch.nn.BCEWithLogitsLoss()
```
This combines sigmoid activation and binary cross-entropy in a numerically stable way.

The training code also calculates a positive-class weight from the class distribution:

```
Class 0 → Pristine
Class 1 → Fake
```
This helps account for imbalance between the two classes.

---

# 15. Optimizers
The training implementation supports:

```
SGD
AdamW
Adam
```
The optimizer is selected through the YAML configuration.

The included configuration uses:

```
optimizer: 'SGD'
```

---

# 16. Learning Rate Scheduling
The code supports learning-rate scheduling.

The supplied configuration uses:

```
scheduler: 'cosinelr'
```
The implementation uses `CosineLRScheduler` from `timm`.

The goal is to change the learning rate throughout training rather than maintaining a single constant value.

---

# 17. Validation Loop
After every training epoch, the model is evaluated on the validation set.

```
Training Epoch
      │
      ▼
Validation Set
      │
      ▼
Model Prediction
      │
      ▼
Validation Loss
      │
      ├── Accuracy
      └── F1 Score
      │
      ▼
Compare With Previous Epoch
      │
      ▼
Checkpoint / Patience Tracking
```
The training script calculates:

```
Validation Loss
Validation Accuracy
Validation F1
```
and uses validation loss to determine whether a checkpoint should be saved.

---

# 18. Early Stopping
The training pipeline keeps track of whether validation loss improves.

```
Validation loss improves
        ↓
Reset patience
        ↓
Continue training
```
If validation loss repeatedly fails to improve:

```
No improvement
     ↓
Patience increases
     ↓
Patience reaches limit
     ↓
Training stops
```
The parameter is controlled using:

```
--patience
```
The default in the training script is:

```
10
```

---

# 19. Checkpointing
When validation loss improves, the model state is saved.

Depending on the selected model, the checkpoint name identifies the architecture.

Examples include:

```
CrossViT_checkpoint...
Resnet50_checkpoint...
Swin_checkpoint...
```
The output directory can be specified using:

```
--models_output_path
```

---

# 20. Custom Validation Dataset
The project contains:

```
construct_validation_set.py
```
for constructing the validation set described by the original DFAD workflow.

The research paper describes a custom validation set consisting of:

```
2,500 pristine images
+
2,500 fake images
=
5,000 total validation images
```
The pristine images were sourced from:

```
Wikimedia
MSCOCO
Flickr
```
The fake images were generated using methods including:

```
StyleGAN
StyleGAN2
ProGAN
RelGAN
GLIDE
Stable Diffusion
```
The key idea is **cross-generator evaluation**.

Instead of testing only on one type of synthetic image, the detector is exposed to multiple generation families.

---

# 21. Validation Workflow

```
Pristine Sources
      │
      ├────────────┐
      │            │
      ▼            ▼
Wikimedia       MSCOCO / Flickr
      │            │
      └──────┬─────┘
             ▼
       Pristine Set
             │
             │
             ├───────────────────────┐
             │                       │
             ▼                       ▼
       GAN Generators          Diffusion Models
             │                       │
             └──────────┬────────────┘
                        ▼
                   Fake Set
                        │
                ┌───────┴────────┐
                ▼                ▼
             Pristine           Fake
                │                │
                └───────┬────────┘
                        ▼
                Validation Dataset
```
This structure is useful for studying generalization to unseen generation techniques.

---

# 22. Evaluation / Inference Pipeline
The main evaluation entry point is:

```
test.py
```
The basic inference process is:

```
Input Image
     │
     ▼
Resize + Padding
     │
     ▼
Trained Model
     │
     ▼
Raw Logit
     │
     ▼
Sigmoid
     │
     ▼
Fake Probability
     │
     ▼
Decision Threshold
     │
     ├───────────────┐
     ▼               ▼
PRISTINE           FAKE
```
The default decision threshold is:

```
0.5
```
and can be modified with:

```
--threshold
```

---

# 23. Example of Prediction Logic
Conceptually:

```
Model output = 0.13
        ↓
Fake probability = 0.13
        ↓
0.13 < 0.50
        ↓
PRISTINE
```
Another example:

```
Model output = 0.87
        ↓
Fake probability = 0.87
        ↓
0.87 ≥ 0.50
        ↓
FAKE
```
The score should be interpreted as a model prediction, not as absolute proof of authenticity.

---

# 24. Output Format
Predictions are written to a JSON file.

Conceptually:

```
{
  "image_001.jpg": 0.87,
  "image_002.jpg": 0.13
}
```
The output location is controlled by:

```
--output_path
```

---

# 25. Evaluation Metrics
When ground-truth labels are supplied, the evaluation script can calculate:

```
Accuracy
F1 Score
ROC Curve
AUC
```
These metrics provide different views of model performance.

### Accuracy
Measures the overall fraction of correctly classified samples.

### F1 Score
Balances precision and recall and is useful when class distributions are not perfectly balanced.

### ROC / AUC
Measures ranking behaviour across different classification thresholds.

---

# 26. Ensemble Prediction
The evaluation script supports ensemble inference.

Instead of relying on one model:

```
                 Input Image
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Model 1     Model 2      Model 3
          │           │           │
          ▼           ▼           ▼
        Score       Score       Score
          │           │           │
          └───────────┼───────────┘
                      ▼
              Average Probability
                      │
                      ▼
               Final Prediction
```
The ensemble option is enabled using:

```
--ensemble
```
and requires:

```
--model1_weights
--model2_weights
--model3_weights
```
The implementation averages the predicted probabilities before applying the decision logic.

---

# 27. Error Analysis
The project also includes an optional error-analysis mode.

This is particularly useful when the test dataset contains the known generation method.

The workflow is:

```
Test Sample
     │
     ▼
Model Prediction
     │
     ▼
Correct?
   /     \
 YES      NO
          │
          ▼
 Read Generation Method
          │
          ▼
 Aggregate Error
```
This allows researchers to investigate questions such as:

```
Which generator causes the most errors?
Which manipulation family is difficult?
Does the detector generalize equally well across generators?
```
This is more informative than looking only at overall accuracy.

Enable it with:

```
--error_analysis
```

---

# 28. Feature Comparison and t-SNE
The repository also contains:

```
compare_features.py
```
This script uses CLIP image embeddings and t-SNE to visualize feature distributions.

The workflow is approximately:

```
Image
  │
  ▼
CLIP Encoder
  │
  ▼
Image Embedding
  │
  ▼
High-dimensional Feature Space
  │
  ▼
t-SNE
  │
  ▼
2D Visualization
```
It can compare embeddings from:

```
Fake training samples
Real training samples
Test samples
```
The purpose is to visually investigate whether these groups occupy distinguishable regions in feature space.

This is an exploratory analysis tool rather than the main deepfake classifier.

---

# 29. Augmentation Visualization
The repository includes:

```
augment.py
```
which repeatedly applies the augmentation pipeline to an example image.

Conceptually:

```
Original Image
      │
      ├── Augmentation 1
      ├── Augmentation 2
      ├── Augmentation 3
      ├── Augmentation 4
      ├── ...
      └── Augmentation 200
                │
                ▼
       Visualization Grid
                +
              GIF
```
This allows the transformation pipeline to be visually inspected before training.

The generated artifacts include:

```
augmented_images.png
augmented_images.gif
```

---

# 30. Google Colab Workflow
The repository contains:

```
colab_setup.ipynb
```
which provides a Google Colab-oriented setup.

The notebook covers:

```
Mount Google Drive
       ↓
Clone / Upload Repository
       ↓
Install Dependencies
       ↓
Prepare Dataset
       ↓
Train Model
       ↓
Evaluate Model
```
The main dependencies listed by the notebook include:

```
pip install timm albumentations progress datasets torchsr PyYAML scikit-learn
```
For Colab, a lower number of DataLoader workers such as:

```
2–4 workers
```
is more appropriate than the high worker defaults used in larger training environments.

---

# 31. Training Command
Example:

```
python train.py \
  --config configs/architecture.yaml \
  --training_csv /path/to/training_set.csv \
  --validation_csv /path/to/validation_set.csv \
  --models_output_path models/ \
  --model 2 \
  --num_epochs 60 \
  --workers 2
```
Model identifiers:

```
0 → Cross-Efficient ViT
1 → ResNet-50
2 → Swin Transformer
```

---

# 32. Evaluation Command
Example:

```
python test.py \
  --config configs/architecture.yaml \
  --test_folder /path/to/test_set/ \
  --model1_weights /path/to/checkpoint \
  --model 2 \
  --workers 2
```

---

# 33. Evaluation With Ground-Truth Labels
To evaluate performance rather than only generate predictions, provide a labeled CSV:

```
python test.py \
  --config configs/architecture.yaml \
  --correct_labels_csv /path/to/test.csv \
  --model1_weights /path/to/checkpoint \
  --model 2 \
  --workers 2
```
The labeled CSV should provide the path and label information required by the evaluation code.

---

# 34. Ensemble Evaluation
Example:

```
python test.py \
  --config configs/architecture.yaml \
  --test_folder /path/to/test_set/ \
  --ensemble \
  --model1_weights /path/to/model1 \
  --model2_weights /path/to/model2 \
  --model3_weights /path/to/model3 \
  --workers 2
```

---

# 35. Configuration
Configuration files are located inside:

```
configs/
```
The YAML configuration controls training and architecture parameters.

Example:

```
training:
  lr: 0.01
  weight-decay: 0.0000001
  bs: 64
  optimizer: 'SGD'
  scheduler: 'cosinelr'

test:
  bs: 1

model:
  image-size: 224
  num-classes: 1
```
The Cross-Efficient ViT configuration additionally specifies:

```
Model depth
Patch sizes
Attention heads
Embedding dimensions
MLP dimensions
Cross-attention depth
Dropout
```

---

# 36. Command-Line Arguments

## `train.py`
| Argument | Description |
|---|---|
| `--config` | YAML configuration file |
| `--model` | Model architecture |
| `--training_csv` | Training dataset CSV |
| `--validation_csv` | Validation dataset CSV |
| `--num_epochs` | Number of training epochs |
| `--workers` | DataLoader worker count |
| `--resume` | Resume from checkpoint |
| `--models_output_path` | Checkpoint output directory |
| `--max_images` | Maximum samples to use |
| `--pre_load_images` | Pre-load images into memory |
| `--image_mode` | Select preprocessing/frequency mode |
| `--efficient_net` | EfficientNet option for Cross-Efficient ViT |
| `--patience` | Validation-loss patience |
| `--gpu_id` | GPU identifier |
| `--random_state` | Random seed |

## `test.py`
| Argument | Description |
|---|---|
| `--test_folder` | Folder containing test images |
| `--correct_labels_csv` | Ground-truth CSV |
| `--output_path` | JSON prediction output |
| `--max_images` | Limit number of test samples |
| `--ensemble` | Enable ensemble inference |
| `--error_analysis` | Analyze errors by generation method |
| `--config` | YAML configuration |
| `--gpu_id` | GPU identifier |
| `--model` | Model architecture |
| `--threshold` | Fake/pristine threshold |
| `--model1_weights` | First checkpoint |
| `--model2_weights` | Second checkpoint |
| `--model3_weights` | Third checkpoint |

---

# 37. Repository Structure

```
Deepfake-Image-and-video-analyzer/
│
├── train.py
│   └── Main model training pipeline
│
├── test.py
│   └── Inference, evaluation, ensemble prediction
│
├── deepfakes_dataset.py
│   └── Dataset loading, preprocessing and augmentation
│
├── augment.py
│   └── Augmentation visualization utility
│
├── construct_training_set_csv.py
│   └── Training dataset CSV construction
│
├── construct_validation_set.py
│   └── Custom validation dataset construction
│
├── compare_features.py
│   └── CLIP feature extraction and t-SNE analysis
│
├── configs/
│   ├── architecture.yaml
│   ├── architecture2.yaml
│   └── architecture copy.yaml
│
├── cross-efficient-vit/
│   ├── configs/
│   ├── deepfakes_dataset.py
│   └── efficient_net/
│
├── colab_setup.ipynb
│   └── Google Colab setup and execution workflow
│
├── Research-paper-deepfake-model-by-sm.pdf
│   └── Associated research paper
│
├── .gitignore
│
└── README.md
```

---

# 38. Research Paper
The repository includes:

```
Research-paper-deepfake-model-by-sm.pdf
```
This document provides the research background and methodology associated with the DFAD 2023 implementation, including:

- Data augmentation
- Model architecture
- Validation-set construction
- Training methodology
- Evaluation methodology
- Related research
The paper should be treated as the **research reference for the implementation**.

The presence of the paper in this repository does **not** by itself mean that every reported result has been independently reproduced here.

---

# 39. Important Video Analysis Note
The repository name contains:

```
Deepfake-Image-and-video-analyzer
```
However, the currently documented root pipeline is primarily an **image classification pipeline**.

The available training and evaluation code processes individual images. The existing implementation does not provide a complete end-to-end video inference layer that:

```
Video
  ↓
Decode frames
  ↓
Sample frames
  ↓
Detect / crop faces
  ↓
Preprocess frames
  ↓
Run deepfake classifier
  ↓
Aggregate temporal predictions
  ↓
Generate video-level result
```
Therefore, the repository should not claim that a complete production video detector is already implemented unless such a pipeline is added.

A proper video extension would look like:

```
                    INPUT VIDEO
                         │
                         ▼
                  Frame Extraction
                         │
                         ▼
                  Frame Sampling
                         │
                         ▼
                 Face / ROI Detection
                         │
                         ▼
                Frame Preprocessing
                         │
                         ▼
                Deepfake Classifier
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
        Frame Scores             Confidence
             │
             ▼
        Temporal Aggregation
             │
             ▼
        Video-level Score
             │
             ▼
        FAKE / PRISTINE
```
The current image classifier can serve as the frame-level model for such an extension.

---

# 40. Why Generalization Matters
One of the biggest challenges in deepfake detection is **dataset shift**.

A model can perform well when the test data comes from the same distribution as the training data, but performance can decrease on:

```
New Generators
New Compression
New Resolutions
New Editing Pipelines
New Diffusion Models
New Face Manipulation Methods
```
That is why the custom validation workflow is important.

The project is interested not only in:

> “Can the model classify these images?”
but also:

> “Does the model remain useful when the source of the fake changes?”

---

# 41. Research-Oriented Experimental Flow
The methodology can therefore be summarized as:

```
                  DATA COLLECTION
                        │
                        ▼
                DATASET PREPARATION
                        │
                        ▼
                TRAIN/VALIDATION SPLIT
                        │
                        ▼
             ROBUST IMAGE AUGMENTATION
                        │
                        ▼
                  MODEL TRAINING
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
          CrossViT    ResNet      Swin
             │          │          │
             └──────────┼──────────┘
                        ▼
                 VALIDATION
                        │
                        ▼
             CHECKPOINT SELECTION
                        │
                        ▼
                 TEST / INFERENCE
                        │
               ┌────────┼────────┐
               ▼        ▼        ▼
            Accuracy    F1       AUC
                        │
                        ▼
                 ERROR ANALYSIS
                        │
                        ▼
                GENERALIZATION
```

---

# 42. Strengths of the Approach
The main strengths of the project are:

### Robust augmentation
The model is exposed to a broad collection of realistic image transformations.

### Multiple architectures
The repository allows comparison between:

```
Cross-Efficient ViT
ResNet-50
Swin Transformer
```

### Frequency-domain experimentation
FFT and DCT processing provide an additional way to expose or manipulate image-frequency information.

### Cross-generator validation
The validation workflow includes multiple GAN and diffusion generation methods.

### Error analysis
Mistakes can be examined by generation method instead of relying only on aggregate metrics.

### Ensemble inference
Multiple model predictions can be combined to produce a final score.

---

# 43. Limitations
Deepfake detection is not equivalent to proving whether an image is authentic.

A prediction from a machine-learning model can be wrong, especially when the input distribution differs from the training data.

Important limitations include:

```
Dataset shift
Unseen generators
Compression artifacts
Image quality
Resolution changes
Adversarial manipulation
Domain-specific distributions
```
Therefore:

> **The model's prediction should be treated as a machine-learning assessment, not as absolute forensic proof.**

---

# 44. Current Project Scope

### Implemented / represented in the repository

- Binary deepfake/pristine image classification
- Cross-Efficient ViT
- ResNet-50
- Swin Transformer
- Strong image augmentation
- FFT processing
- DCT processing
- Custom validation-set construction
- Model training
- Validation
- Checkpoint saving
- Single-model inference
- Ensemble inference
- Accuracy / F1 / AUC evaluation
- Generation-method error analysis
- CLIP feature visualization
- t-SNE analysis
- Colab setup workflow

### Not presented as a completed implementation

- End-to-end video decoding pipeline
- Temporal modeling
- Frame-to-video score aggregation
- Production forensic certification
- Independently reproduced DFAD-2023 benchmark results

---

# 45. References
The implementation is based on the DFAD 2023 research work and its cited literature.

### Primary research
**Coccomini, Davide Alessandro; Amato, Giuseppe; Falchi, Fabrizio; Gennaro, Claudio.**

DFAD 2023 — Challenge on DeepFake Analysis and Detection.

ISTI-CNR.

**1st place at DFAD 2023.**

### Related work

1. Coccomini et al. — *MINTIME: Multi-Identity Size-Invariant Video Deepfake Detection*.
2. Coccomini et al. — *Detecting Images Generated by Diffusers*.
3. Coccomini et al. — *On the Generalization of Deep Learning Models in Video Deepfake Detection*.
4. Giudice et al. — *Fighting Deepfakes by Detecting GAN DCT Anomalies*.
5. Liu et al. — *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*.
6. Coccomini et al. — *Cross-Forgery Analysis of Vision Transformers and CNNs for Deepfake Image Detection*.
7. Coccomini et al. — *Combining EfficientNet and Vision Transformers for Video Deepfake Detection*.
8. Guarnera et al. — *The Face Deepfake Detection Challenge*.
For the full bibliography and methodology, see:

```
Research-paper-deepfake-model-by-sm.pdf
```

---

# 46. Repository
GitHub:

[https://github.com/ShlokMishra01/Deepfake-Image-and-video-analyzer](https://github.com/ShlokMishra01/Deepfake-Image-and-video-analyzer)

---

# 47. Final Takeaway
The project is fundamentally a **robust deepfake image-classification research pipeline**.

Its workflow is:

```
                IMAGE
                  │
                  ▼
            PREPROCESSING
                  │
                  ▼
        ROBUST AUGMENTATION
                  │
                  ▼
          VISION TRANSFORMER /
             CNN MODEL
                  │
                  ▼
          FAKE PROBABILITY
                  │
                  ▼
             THRESHOLD
                  │
             ┌────┴────┐
             ▼         ▼
          PRISTINE     FAKE
```
The larger research objective is to move beyond simple classification and investigate whether deepfake detectors can **generalize across different generation methods and image conditions**.

The repository provides the building blocks required for that experimentation while keeping the original DFAD 2023 research methodology and attribution explicit.

# Facial Recognition with a Convolutional Neural Network

**20-class facial identity classification using TensorFlow/Keras on the ORL Faces dataset.** The trained CNN achieved **92.5% accuracy (148/160 correct predictions)** on the notebook's held-out test split.

## Overview

This project explores supervised facial identity classification from grayscale face images. A convolutional neural network (CNN) is trained to distinguish among **20 identities** in the supplied `ORL_faces.npz` dataset. The workflow covers data inspection, image preprocessing, model design, training, evaluation, and model export.

> **Scope:** This is a closed-set image classifier for identities represented in the training data. It does **not** implement face detection, verification of unseen identities, or a deployed inference API.

## Results

| Metric | Result |
| --- | ---: |
| Test accuracy | **92.5%** |
| Correct test predictions | **148 / 160** |
| Number of classes | 20 |
| Training images before validation split | 240 |
| Training images after validation split | 228 |
| Validation images | 12 |
| Test images | 160 |
| Input dimensions | 112 × 92 × 1 |
| Trainable parameters | 52,500,114 |

The notebook also generates a **confusion matrix** and a **per-class classification report**. The validation split is very small (12 images), so validation accuracy should be interpreted cautiously. Results reflect the supplied split rather than an independently established benchmark.

## Dataset and Preprocessing

The notebook loads `ORL_faces.npz` from Google Drive. The archive contains `trainX`, `trainY`, `testX`, and `testY` arrays. The notebook uses a subset with **20 identity classes**.

1. Load the training and test arrays from the `.npz` archive.
2. Reshape flattened 10,304-pixel images to **112 × 92** grayscale images.
3. Add a channel dimension to produce tensors of shape **(112, 92, 1)**.
4. Normalize pixel intensities from **0–255** to **0–1**.
5. Split the supplied training data into training and validation sets (`test_size=0.05`, `random_state=42`).

**Data requirement:** `ORL_faces.npz` is referenced by the notebook but is not embedded in it. Obtain an appropriately licensed copy of the dataset and update the notebook's Google Drive path before running.

## CNN Architecture

```text
Input image: 112 × 92 × 1
        │
Conv2D: 36 filters, 7×7, ReLU
        │
MaxPooling2D: 2×2
        │
Conv2D: 54 filters, 5×5, ReLU
        │
MaxPooling2D: 2×2
        │
Flatten
        │
Dense: 2,024 units, ReLU → Dropout (0.5)
        │
Dense: 1,024 units, ReLU → Dropout (0.5)
        │
Dense:   512 units, ReLU → Dropout (0.5)
        │
Dense:    20 units, Softmax
        │
Predicted identity (class 0–19)
```

**Training configuration:** Adam optimizer (`learning_rate=0.001` in the original notebook), sparse categorical cross-entropy loss, **250 epochs**, and **batch size 512**. Because the training split contains only 228 images, this batch size results in a single optimization step per epoch.

## Run the Notebook

1. Open `Facial_Recognition_Final_1(1).ipynb` in **Google Colab**.
2. Make the `ORL_faces.npz` dataset available in Google Drive.
3. Update the dataset path in the notebook if needed:

   ```python
   data = np.load('/content/gdrive/My Drive/Facial_Recognition/ORL_faces.npz')
   ```

4. Install compatible versions of the dependencies: **TensorFlow/Keras**, **NumPy**, **Matplotlib**, and **scikit-learn**.
5. Execute the notebook cells in order to preprocess images, train the CNN, evaluate predictions, and export the model.

> **Compatibility note:** The notebook was written against an older TensorFlow/Keras API. `Adam(lr=...)`, `model.predict_classes(...)`, and the legacy SavedModel export call may require updates on current versions. For example, use `Adam(learning_rate=0.001)` and `np.argmax(model.predict(X_test), axis=1)`.

## Model Evaluation

The notebook evaluates the model on the held-out test split and calculates:

- Overall classification accuracy
- Confusion matrix
- Precision, recall, and F1-score by identity (via `classification_report`)

Recorded test accuracy:

```text
accuracy:0.925
```

This corresponds to **148 correct classifications out of 160 test images**.

## Model Export

The original notebook saves the trained model using TensorFlow's **SavedModel** format:

```python
CNN_model.save('saved_model/my_FR_model')
```

This is the original code and may require adaptation for newer Keras versions. For example, modern Keras supports `.keras` model files and has separate SavedModel export methods.

**Note:** The notebook demonstrates model saving but does not include a REST API, Postman workflow, or production deployment.

## Repository Contents

```text
facial-recognition-cnn-tensorflow/
├── Facial_Recognition_Final_1(1).ipynb  # Training and evaluation notebook
└── README.md                            # Project documentation
```

The dataset and exported model are external to the notebook; include them only when you have the rights and a reason to distribute them. Avoid committing private facial imagery or credentials.

## Limitations and Future Improvements

- **Small dataset:** 228 training images across 20 identities; generalization is uncertain.
- **Tiny validation split:** Only 12 validation images; use more robust validation and consider stratification.
- **Large model:** More than 52 million parameters creates substantial overfitting risk relative to dataset size.
- **Reproducibility:** Pin dependency versions and random seeds; document the exact dataset source and split.
- **Modernization:** Update deprecated TensorFlow/Keras calls and consider checkpointing or early stopping.
- **Comparison:** Benchmark against a smaller CNN or transfer-learning model, using appropriate identity-disjoint evaluation for any claims about generalization to unseen people.
- **Responsible use:** Facial recognition involves sensitive biometric information. This academic demonstration is not validated for security, surveillance, or consequential identity decisions.

## Tech Stack

**Python · TensorFlow · Keras · NumPy · Matplotlib · scikit-learn · Google Colab**

## Project Status

**Completed experimental notebook.** The documented metrics come from the saved notebook outputs; no independent rerun or production deployment is claimed.

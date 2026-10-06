# Deepfake Detection Using InceptionV3

An end-to-end Deep Learning and Computer Vision project for detecting manipulated and synthetic deepfake media (images/videos) utilizing Transfer Learning with **InceptionV3**.

---

## 📌 Project Overview
This project develops an accurate deep learning classification pipeline to distinguish authentic media from AI-generated/manipulated deepfake media. Leveraging the InceptionV3 architecture pre-trained on ImageNet, the model extracts high-level spatial feature representations to classify frames as either **Real** or **Fake**.

---

## 🛠️ Key Technical Features

- **Data Preprocessing & Augmentation:**
  - Standardized frame extraction, normalization, and real-time data augmentation using Keras `ImageDataGenerator` (rotation, zoom, horizontal flips) to minimize overfitting.

- **Transfer Learning Architecture:**
  - Base Model: **InceptionV3** initialized with pre-trained ImageNet weights.
  - Custom Classification Head:
    - Global Average Pooling 2D
    - Fully Connected (Dense) layers with Batch Normalization
    - Dropout layers (0.3 - 0.5) for regularization
    - Sigmoid activation output layer for binary probability classification.

- **Training Callbacks & Optimization:**
  - Implemented `EarlyStopping` to halt training at optimal validation loss.
  - `ModelCheckpoint` to persist best-performing model weights.
  - `ReduceLROnPlateau` for adaptive learning rate scheduling.

- **Evaluation Metrics:**
  - Evaluated via Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
  - Achieved ~81% validation accuracy with a balanced F1-score of 0.74 on unseen test splits.

---

## 📂 Repository Structure
```
├── DFVD.ipynb               # Deepfake video frame extraction & initial experiments
├── Preprocessing.ipynb      # Frame normalization, data cleaning, and augmentation
├── Model_Training.ipynb     # InceptionV3 transfer learning model architecture & training
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow / Keras
- OpenCV (`cv2`)
- NumPy, Pandas, Matplotlib, Scikit-Learn

### Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/lavyadav128/DeepFake-Detection.git
   cd DeepFake-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
   ```

3. **Run Notebooks:**
   Open Jupyter Notebook or Google Colab and run `Preprocessing.ipynb` followed by `Model_Training.ipynb`.

---

## 👤 Author
- **Lav Kumar Yadav**
- GitHub: [@lavyadav128](https://github.com/lavyadav128)
- LinkedIn: [Lav Kumar Yadav](https://www.linkedin.com/in/lav-yadav-90476981)

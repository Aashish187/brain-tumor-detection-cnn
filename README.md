# Brain Tumor Detection using CNN (MRI Images)

This project develops an automated **brain tumor detection system** using **Convolutional Neural Networks (CNNs)** to classify brain MRI images as **Tumor** or **Non-Tumor** for medical assistance.

---

## 🚀 Project Overview

- **Task**: Binary image classification (Tumor vs Non-Tumor) on brain MRI scans.  
- **Approach**: Custom CNN built with **TensorFlow/Keras**, trained on a **combined dataset from 4 Kaggle sources** to improve robustness and reduce overfitting.  
- **Result**: Achieved around **91–92% validation accuracy** using training callbacks such as EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.

---

## 📂 Datasets

This project merges four publicly available MRI datasets into a single unified dataset:

1. **Brain Tumor Detection** – `ahmedhamada0/brain-tumor-detection`  
   - https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection  

2. **Brain MRI Images for Brain Tumor Detection** – `navoneel/brain-mri-images-for-brain-tumor-detection`  
   - https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection  

3. **Brain Tumor MRI Dataset** – `masoudnickparvar/brain-tumor-mri-dataset`  
   - https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset  

4. **Brain Tumor MRI Data** – `tombackert/brain-tumor-mri-data`  
   - https://www.kaggle.com/datasets/tombackert/brain-tumor-mri-data  

> **Note**: The raw image data is **not included** in this repository due to size and licensing. Please download it yourself from Kaggle.

---

## 🧠 Methodology

### 1. Data Preparation

- Download all four Kaggle datasets using the **Kaggle API** in Colab.  
- Extract and move images into a unified directory structure, separating **Tumor** and **Non-Tumor**.  
- Apply preprocessing:
  - Resize images to fixed size (128x128 or 224x224).  
  - Normalize pixel values.  
  - Use `ImageDataGenerator` for **data augmentation** (rotation, shifts, zoom, flips, brightness changes).  

### 2. Model Architecture

The model uses a **CNN with 4 Conv2D blocks** combined with:

- `Conv2D` + `BatchNormalization` + `MaxPooling2D` layers for feature extraction.  
- `GlobalAveragePooling2D` to reduce dimensionality.  
- `Dense` layers with `Dropout` (0.5) to prevent overfitting.  
- Final layer: `Dense(1, activation='sigmoid')` for binary classification.  
- **Optimizer**: Adam (learning rate 1e-4).  
- **Loss**: Binary cross-entropy.  

### 3. Training Strategy

- Train/validation/test split using `train_test_split`.  
- Callbacks for robust training:
  - `EarlyStopping`: Stop when validation accuracy plateaus.  
  - `ReduceLROnPlateau`: Reduce learning rate when loss stops improving.  
  - `ModelCheckpoint`: Save the best model as `best_brain_tumor_model.keras`.  

---

## 📊 Results

- **Validation Accuracy**: ~91–92% (binary classification, Tumor vs Non-Tumor).  
- The model shows strong performance on unseen validation data when trained on the combined dataset with data augmentation and proper regularization.  

---

## 🧪 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Aashish187/brain-tumor-detection-cnn.git
cd brain-tumor-detection-cnn
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Datasets

- Create a Kaggle account and download the four datasets listed above.  
- Organize them in an `all_datasets/` folder or modify the notebook paths accordingly.  

### 4. Run the Notebook

- Open `notebooks/Brain_Tumor_Detection_CNN.ipynb` in Jupyter or Google Colab.  
- Follow the cells in order:
  1. Import libraries and configure GPU.  
  2. Download and organize datasets.  
  3. Preprocess images and create train/validation/test sets.  
  4. Define and train the CNN.  
  5. Evaluate and visualize results.  

---

## 🔧 Technologies Used

- **Language**: Python  
- **Frameworks**: TensorFlow, Keras  
- **Libraries**: NumPy, Pandas, OpenCV, Matplotlib, Seaborn, Scikit-learn  
- **Platform**: Google Colab (GPU enabled)  

---

## 📌 Future Work

- Add Grad-CAM visualizations for model interpretability.  
- Extend to multi-class classification (e.g., different tumor types).  
- Deploy as a web application (Streamlit or Flask).  
- Test on additional medical imaging datasets.  

---

## 📜 Acknowledgements

Thanks to the authors of the public Kaggle datasets used in this project for making their MRI collections available for research and educational purposes.

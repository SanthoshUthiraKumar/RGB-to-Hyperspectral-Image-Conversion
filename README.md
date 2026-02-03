# **Deep Learning-Based Spectral Reconstruction & Crop Classification**
## **1. Project Overview**
This project provides a complete software pipeline to convert standard RGB images (3 bands) into scientific-grade Hyperspectral Data Cubes (31+ bands). It subsequently uses this reconstructed data to classify crop types and growth stages without requiring expensive physical hyperspectral cameras.
### **Key Capabilities**
- **Spectral Super-Resolution:** Uses a Conditional GAN (Pix2HS/ResNet) to estimate spectral signatures from 400nm to 1000nm.
- **Robust Pre-processing:** Features a Patch-Based training system to ignore black backgrounds and focus on leaf texture.
- **Scientific Post-Processing:** Automatically handles unit scaling (0-1 vs 0-100%), background masking, and spectral resampling.
- **Multi-Output Classification:** Uses XGBoost to predict both **Crop Name** (e.g., Corn, Rice) and **Growth Stage** (e.g., Vegetative, Critical).
## **2. System Architecture**
The workflow consists of three distinct stages:

1. **Reconstruction (RGB** ![ref1] **.MAT):**
   1. Input: Standard JPEG/PNG Crop Image.
   1. Model: ResNet-based Generator + PatchGAN Discriminator.
   1. Output: Hyperspectral Cube (Height ![ref2] Width ![ref2] Bands).
1. **Flattening ( .MAT** ![ref1] **.CSV):**
   1. Process: Extracts the leaf area (ignoring background), calculates the mean spectral signature, and interpolates values to match laboratory sensor standards.
1. **Classification (.CSV** ![ref1] **Prediction):**
   1. Model: XGBoost Classifier trained on ground-truth spectral libraries.
   1. Output: "Corn", "Soybean", "Rice", etc.
## **3. Data Sources & Modifications**
This project relies on two specific datasets, modified to ensure compatibility between the AI generation (Vis-NIR) and the scientific ground truth (Vis-SWIR).
### **3.1. Training Data: Agro-HSR\_v1**
- **Source:** A specialized agricultural hyperspectral dataset containing crops like Sweet Potato and Maize.
- **Role:** Used to train the GAN (Pix2HS) to learn the texture-to-spectrum mapping.
- **Our Modifications:**
  - **Patch-Based Loading:** The raw dataset contains images with significant black backgrounds (value 0). Standard training caused the model to bias towards "dark/muddy" outputs. We modified the data loader to extract random **64x64 patches** strictly from valid leaf areas, ignoring the black void entirely.
### **3.2. Ground Truth Library: GHISACONUS\_2008\_001\_speclib\_updated**
- **Source:** The Global Hyperspectral Imaging Spectral-library of Agricultural crops (GHISA-CONUS) from USGS/NASA.
- **Role:** Acts as the "Teacher" for the classification model, providing the true spectral curves for Corn, Soybean, Rice, Cotton, etc.
- **Our Modifications:**
  - **Spectral Truncation:** The original file includes Short-Wave Infrared (SWIR) bands up to 2345nm. Since our AI model (trained on Agro-HSR) only outputs up to 1000nm, we filtered the library to **only use common bands (400nm - 1000nm)** during training.
  - **Header Standardization:** We aligned the column headers (e.g., X437, X447) to ensure the AI-generated CSV matches the library's format exactly for the XGBoost classifier.
## **4. Installation & Requirements**
### **Hardware**
- **GPU:** NVIDIA GPU (4GB+ VRAM) recommended for training.
- **RAM:** 16GB System RAM.
### **Software**
Install the required Python libraries:

pip install torch torchvision numpy pandas scipy h5py opencv-python matplotlib scikit-learn xgboost joblib
## **5. Directory Structure (Crucial)**
Ensure your project folder is organized exactly as follows for the scripts to run without errors:

Project\_Root/\
│\
├── dataset/                  <-- Put TRAINING .mat files here (Agro-HSR\_v1)\
├── input\_images/             <-- Put TEST RGB images (.jpg) here\
├── output\_mats/              <-- Generated Hyperspectral cubes save here\
├── checkpoints/              <-- Model weights (.pth) save here\
│\
├── models.py                 <-- GAN Architecture (ResNet + Discriminator)\
├── train.py                  <-- Patch-Based Training Script\
├── inference\_tiled.py        <-- Script to convert new Images to Cubes\
├── mat\_to\_csv\_resampled.py   <-- Script to flatten Cubes to CSV\
├── classify\_crops.py         <-- Script to predict Crop Name\
├── Ghisaconus\_2008\_001\_speclib\_updated.csv <-- The Lab Data\

## **6. Usage Guide (Step-by-Step)**
### **Step 1: Train the Spectral Reconstruction Model**
This trains the AI to understand the relationship between RGB colors and Spectral curves.

- **Input:** .mat files in the dataset/ folder.
- **Command:**\
  python train.py
- **Output:** Saves netG\_patch\_final.pth in the checkpoints/ folder.
- *Note: Takes ~2 hours on a GPU, or several days on a CPU.*
### **Step 2: Convert RGB Images to Hyperspectral**
Takes a normal photo and generates the spectral data cube.

- **Input:** Images inside input\_images/.
- **Command:**\
  python inference\_tiled.py
- **Output:** Generates .mat files in output\_mats/.
### **Step 3: Flatten to Spectral Library (CSV)**
Converts the 3D image data into a single spectral signature row, matching laboratory standards.

- **Logic:** Applies background masking (threshold > 0.05), scales units (x100), and resamples bands.
- **Command:**\
  python mat\_to\_csv\_resampled.py
- **Output:** Creates Final\_Dataset\_Lab\_Matched.csv.
### **Step 4: Classify the Crop**
Predicts the crop name based on the generated spectral signature.

- **Prerequisite:** Ensure Ghisaconus\_2008\_001\_speclib\_updated.csv is in the root folder.
- **Command:**\
  python classify\_crops.py
- **Output:** Prints the predicted Crop Name and Stage to the console and saves Final\_Crop\_Predictions.csv.
## **7. Troubleshooting Common Issues**
**Issue 1: The output image is just black and brown blocks.**

- **Cause:** The model learned the "Black Background" bias from the training data.
- **Fix:** Ensure you are using the **Patch-Based** train.py. This forces the model to look at leaf texture only.

**Issue 2: The predicted spectral values are tiny (e.g., 0.04 instead of 16.0).**

- **Cause:** Deep Learning outputs 0-1 range, and black backgrounds dilute the average.
- **Fix:** Run mat\_to\_csv\_resampled.py. It includes specific logic to mask out the background and multiply values by 100.

**Issue 3: Cotton is predicted as "Soybean (Mature)".**

- **Cause:** The training dataset (Agro-HSR) lacks Cotton data. The AI interprets white cotton bolls as dead/dry leaves.
- **Fix:** Add real Cotton hyperspectral data to the dataset/ folder and retrain.

**Issue 4: "Dimension Mismatch" errors.**

- **Cause:** Your lab data has SWIR bands (up to 2500nm), but the AI only generates up to 1000nm.
- **Fix:** The classify\_crops.py script automatically detects common overlapping bands and ignores the rest. Do not manually delete columns; let the script handle it.

[ref1]: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAYCAYAAAD6S912AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAOxAAADsQBlSsOGwAAARBJREFUOI3tkr1Kw1AAhb9780OgpIaAOJTgRejSxcFFBGvQZ/AhfAsfRXwCH0IXRwd1Kk25XW1IEKHk3utiQeqU1Kn0Ww/n4wwHdmw/XpeOUuq41+v5dV1X66Fsa8vzXDjnjoIgOP+XhdPplDiOfWAcx/FLVVVfGy0EbBiGBaCllNdAsNFCgMVisUyS5AM46/f7h2mavpdl2QAIpVRkjLmRUg7bSJ1zAtgHToQQb8aYW631s/iR7gFRy6GRc+7SOTe21t5prR+BRrSUrAiyLDuVUl54nnc/mUxmq8DvYhsMBgdCiKumaR6Kopj9zroIPd/3h8DnfD5/XQ+73AZrrQGegGWX/h9Go1FIx8vt2Aa+AaYoU9HBibyuAAAAAElFTkSuQmCC
[ref2]: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAYCAYAAADzoH0MAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAOxAAADsQBlSsOGwAAAYVJREFUOI3tUMFKG1EUvfe9GTOSDDEx0cHUvoFkIVk0le4FF3YhFlwI/lbX/YD6FS5cue1OsDCjQZmAMMzMSzDFl3e60RBj6rqCZ3O5nHfOO/cQveP/gux2u0uv8OKRF89Ej5ODIFg1xuzWarUiz/PhvFgp1TPGbJZKpXQ0Gj1MiamTlJaZ1wF8C8MwmBGzUqoHYAfAsuu6dlEC0lr/8X1/QERbALYqlcqV1vpeKfXZWrsrhPjted55HMejWQOejxqG4UcAB9ZaA+BCCLENIHJd9yyKony+GDm3I8uyolqt3jHzF2beA/BLSnkax/EL8bMOZvB0Y42IAmb2HcexC94tTECtVuuDlPKIiBIAJ0KIT5PJZMX3/bgoCvOqQbvdXmPmYwBGCHHS7/evy+XygJm/MvNyvV6/zrLMLDRoNpsVx3EOAawOh8MfSZLkRAStddpoNG6IaJ+IHjqdzm2SJNOTnjoQnuetW2vd8Xj8PU3TYuYTRFF0CeAnEW2kaVr/Vx/veLP4C61zotZvIpUqAAAAAElFTkSuQmCC

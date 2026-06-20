# Fashion Apparel Classification using CNN (Computer Vision)

A production-ready Deep Learning project that designs, trains, and evaluates a **Convolutional Neural Network (CNN)** on the famous **Fashion-MNIST dataset**. The model successfully classifies 10 different categories of clothing and footwear with high precision.

##  Project Workflow & Architecture

### 1. Data Loading & Profiling
- Loads 60,000 training and 10,000 test images structured as `(28, 28, 1)` grayscale matrices.
- Maps numeric labels to descriptive human classes (e.g., T-shirt, Trouser, Ankle Boot).

### 2. CNN Model Architecture (Sequential)
- **Conv2D Layers:** Extracts spatial features using 32 and 64 filters respectively.
- **MaxPooling2D:** Downsamples feature maps to optimize computation.
- **Flatten Layer:** Transforms multi-dimensional matrices into a 1D feature vector.
- **Dense & Dropout:** Applies fully-connected layers with dropout regularizations to mitigate overfitting.

### 3. Training & Performance Evaluation
- Trained over 8 comprehensive epochs, driving loss down to `0.2035`.
- Achieved a stellar **Test Accuracy of 91.24%**.
- Includes a detailed `Classification Report` and a `Confusion Matrix` for evaluating inter-class misclassifications (e.g., distinguishing between Shirts and Coats).

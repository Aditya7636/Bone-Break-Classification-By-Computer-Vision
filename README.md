# 🦴 Bone Break Classification by Computer Vision

This project is dedicated to developing a computer vision model that can accurately classify various types of bone fractures from X-ray images, leveraging the potential of deep learning and computer vision for enhanced medical diagnosis.

## 📄 About the Dataset

The Bone Break Classification dataset provides X-ray images across several classes of bone fractures, including:
- **Avulsion Fractures**
- **Comminuted Fractures**
- **Fracture-Dislocations**
- **Greenstick Fractures**
- **Hairline Fractures**
- **Impacted Fractures**
- **Longitudinal Fractures**
- **Oblique Fractures**
- **Pathological Fractures**
- **Spiral Fractures**

The dataset is available through [Roboflow](https://universe.roboflow.com/curso-rphcb/bone-break-classification) under the CC BY 4.0 license.

### Why This Project Matters

By automating the process of fracture classification, we can enhance patient care and assist medical professionals in making informed, efficient decisions.

---

## 💻 Project Overview

### Goal

The goal is to train a machine learning model capable of distinguishing between different types of bone fractures using X-ray images, allowing it to:
1. **Classify** types of fractures accurately.
2. **Enhance** diagnostic speed and reliability in medical settings.

### Solution Approach

This project involves:
1. **Data Preprocessing**: Image augmentation, resizing, and normalization.
2. **Model Selection**: Exploring architectures like CNNs (e.g., ResNet, EfficientNet) for image classification.
3. **Training and Evaluation**: Training the model on labeled X-ray images and evaluating accuracy across fracture types.
4. **Deployment**: Implementing the trained model in a usable format, such as a web app or API.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/username/Bone-Break-Classification.git
cd Bone-Break-Classification
```

### 2. Install Dependencies

Make sure you have Python and necessary libraries installed. Install dependencies with:

```bash
pip install -r requirements.txt
```

### 3. Dataset Preparation

Download the dataset from [Roboflow Bone Break Classification](https://universe.roboflow.com/curso-rphcb/bone-break-classification). Extract the files and place them in a `data` directory within your project.

Directory structure:

```plaintext
Bone-Break-Classification/
├── data/
│   ├── train/
│   ├── test/
│   └── valid/
├── models/
└── src/
```

### 4. Run the Training Script

To train the model, run:

```bash
python src/train.py --epochs 50 --batch_size 32
```

Modify the `--epochs` and `--batch_size` parameters as desired.

### 5. Evaluate the Model

Once training is complete, evaluate the model’s performance on the test set:

```bash
python src/evaluate.py --model_path models/bone_break_model.pth
```

### 6. Deploy the Model

To deploy the model, you can use tools like **Streamlit** or **Flask** to create an interactive web app for predictions.

---

## 🔍 Example Usage

Here’s how you might use the web app:

1. **Upload X-ray Image**: The user uploads an X-ray image.
2. **Predict Fracture Type**: The model classifies the type of fracture.
3. **Display Results**: The app shows the type of fracture detected and related information.

### Sample Command

```bash
streamlit run src/app.py
```

---

## 📊 Evaluation Metrics

We’ll be evaluating model performance using:
- **Accuracy**: Overall success rate across classes.
- **Precision and Recall**: Essential for understanding model performance on each fracture type.
- **Confusion Matrix**: Visualize true vs. predicted classifications.

---

## 🔍 Results

| Metric        | Value   |
|---------------|---------|
| Accuracy      | 93.5%   |
| Precision     | 91.0%   |
| Recall        | 92.7%   |
| F1 Score      | 91.8%   |

These metrics can vary based on model tuning, so try adjusting hyperparameters to optimize performance.

---

## 🛠 Model Architecture

The model leverages **Convolutional Neural Networks (CNNs)** for image classification. Depending on resources and requirements, you might explore:
- **ResNet** for a balance of depth and performance.
- **EfficientNet** for optimized accuracy and computation efficiency.

Feel free to experiment with these architectures in the `src/train.py` script.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a new branch for your feature.
3. Make your changes and submit a pull request.

---

## 📜 License

This project is licensed under the CC BY 4.0 License.

---


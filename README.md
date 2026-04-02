# 🖼️ Image Captioning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning project that automatically generates natural-language captions for images. A **DenseNet201** CNN extracts rich visual features and an **LSTM** decoder turns them into fluent, descriptive sentences. A Streamlit web app lets you upload any image and see the caption instantly.

![demo](https://github.com/user-attachments/assets/462a6cd4-ac88-47ec-9d4f-6cac47477f7c)

---

## 🚀 Live Demo

> **[▶ Open the app on Streamlit Community Cloud](https://image-captioning-imsujarkhanal.streamlit.app/)**

*(If the app is asleep, click **"Wake app"** and wait a few seconds.)*

---

## 📐 Architecture

```
Image  ──► DenseNet201 (feature extractor) ──► 1920-d embedding
                                                        │
                                                  Dense(256)
                                                        │
                                               ┌────────▼────────┐
Caption tokens ──► Embedding(256) ──►  LSTM(256) ◄─── Concat ───┘
                                                        │
                                                  Dense(128)
                                                        │
                                               Softmax over vocab
```

| Component | Details |
|-----------|---------|
| Feature extractor | DenseNet201 (pre-trained on ImageNet, last FC removed) |
| Image embedding | 1920-d → Dense(256, ReLU) |
| Caption model | Embedding(256) + LSTM(256) + Dropout(0.5) |
| Training data | Flickr8k (~8,000 images, 5 captions each) |
| Optimizer | Adam with ReduceLROnPlateau |

---

## 📁 Project Structure

```
Image-Captioning/
├── models/
│   ├── __init__.py
│   ├── caption_model.py       # LSTM-based caption decoder
│   └── feature_extractor.py   # DenseNet201 feature extractor
├── utils/
│   ├── __init__.py
│   ├── caption_generator.py   # Inference helpers
│   ├── data_generator.py      # Keras Sequence data generator
│   ├── data_preprocessing.py  # Text cleaning & feature extraction
│   └── file_utils.py          # Path validation helpers
├── data/
│   └── flickr8k/              # Dataset directory (not tracked)
│       ├── Images/
│       └── captions.txt
├── app.py                     # Streamlit web application
├── train.py                   # Model training script
├── analysis_exploring.py      # Dataset exploration & visualisation
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/imsujarkhanal/Image-Captioning.git
cd Image-Captioning
```

### 2. Create and activate a virtual environment *(recommended)*

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the [Flickr8k dataset from Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place the files as shown below:

```
data/flickr8k/
├── Images/          # ~8,000 .jpg files
└── captions.txt     # image,caption CSV
```

---

## 🔍 Data Exploration

Visualise sample images with their captions before training:

```bash
python analysis_exploring.py
```

---

## 🏋️ Training

Train the model (saves `model.keras`, `feature_extractor.keras`, and `tokenizer.pkl` in the project root):

```bash
python train.py
```

Training uses early stopping (patience = 5) and learning-rate reduction on plateau, so it stops automatically when the validation loss stops improving.

> 💡 GPU is strongly recommended. On CPU, each epoch can take 30–60 minutes.

---

## 🌐 Running the Streamlit App

Make sure the three model artefacts (`model.keras`, `feature_extractor.keras`, `tokenizer.pkl`) are present in the project root, then run:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser, upload a `.jpg` or `.png` image, and the generated caption will appear above the image.

---

## 📝 Notes

- Ensure `data/flickr8k/` paths in `train.py` and `analysis_exploring.py` match your local setup.
- You can extend this to larger datasets (Flickr30k, MS-COCO) for better accuracy.
- The Flickr8k dataset is sometimes sensitive to model architecture changes — re-train from scratch when modifying the model.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

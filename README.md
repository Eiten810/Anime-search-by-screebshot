# Anime Search by Screenshot

An AI-powered system for identifying anime from screenshots using image embeddings and similarity search.

> Upload an anime screenshot → generate an image embedding → search for visually similar frames → identify the most likely anime.

[🇬🇧 English Documentation](README_ENG.md) · [🇷🇺 Русская документация](README_RUS.md)

---

## Overview

**Anime Search by Screenshot** is a computer vision and image retrieval project designed to identify an anime based on a screenshot.

Instead of treating the task as a traditional image classification problem, the project uses **embedding-based similarity search**.

Each image is transformed into a numerical vector representation called an **embedding**. The embedding of a query screenshot is then compared with embeddings from the dataset to retrieve the most visually similar images.

The project compares different approaches to image representation:

- Gemini Embedding 2 — original image size
- Gemini Embedding 2 — 512×512 images
- ResNet50 embeddings
- ResNet50 trained with Triplet Loss

---

## How It Works

```text
Anime Screenshot
       │
       ▼
┌──────────────────┐
│ Embedding Model  │
└──────────────────┘
       │
       ▼
   Image Embedding
       │
       ▼
┌──────────────────┐
│ Similarity Search │
│      (FAISS)      │
└──────────────────┘
       │
       ▼
Most Similar Images
       │
       ▼
 Predicted Anime
```

### Pipeline

1. A user provides an anime screenshot.
2. The selected model generates an image embedding.
3. The embedding is compared with the database.
4. FAISS retrieves the most similar images.
5. Retrieved results are used to determine the most likely anime.

---

## Models

The project evaluates multiple embedding approaches:

| Model | Image Size | Purpose |
|---|---:|---|
| Gemini Embedding 2 | Original | General-purpose image embeddings |
| Gemini Embedding 2 | 512×512 | Comparison of fixed-size preprocessing |
| ResNet50 | 224×224 | CNN-based visual feature extraction |
| ResNet50 + Triplet Loss | 224×224 | Learning a similarity-oriented embedding space |

---

## Project Structure

```text
Anime-search-by-screenshot/
│
├── config/                 # Project configuration
├── docs/                   # Technical documentation
├── notebooks/              # Experiments, training and evaluation
├── results/                # Evaluation metrics
├── src/                    # Core Python modules
│   ├── config.py
│   ├── data.py
│   ├── embeddings.py
│   ├── infer.py
│   ├── model.py
│   └── train.py
│
├── .env.example            # Environment variable example
├── requirements.txt        # Project dependencies
├── README.md               # Project overview
├── README_ENG.md           # English documentation
└── README_RUS.md           # Russian documentation
```

---

## Technologies

- Python
- PyTorch
- ResNet50
- Gemini Embedding 2
- FAISS
- NumPy
- Scikit-learn
- Jupyter Notebook

---

## Evaluation

The models are evaluated as image retrieval systems using similarity-based metrics.

The evaluation includes:

- Similarity search performance
- Retrieval accuracy
- Comparison between embedding models
- Visualization and analysis of the embedding space

Detailed evaluation results are available in:

```text
results/
notebooks/evaluation.ipynb
```

---

## Documentation

For more detailed information about the project:

- 🇬🇧 [English Documentation](README_ENG.md)
- 🇷🇺 [Русская документация](README_RUS.md)
- 📄 [Technical Documentation](docs/)

---

## Author

**Eiten810**

Data Science / Computer Vision project focused on image embeddings and similarity search.

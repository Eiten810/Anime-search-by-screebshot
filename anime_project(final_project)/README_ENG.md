ENG
# Anime Search by Screenshot

Image Retrieval system for anime identification using image embeddings.

---

## Project Overview

This project implements an **Image Retrieval** system capable of identifying an anime title from a user-uploaded screenshot.

Instead of directly classifying an image, the system converts it into an embedding vector and searches for the most similar images in a database using cosine similarity.

The project compares several embedding extraction approaches and evaluates their retrieval performance using standard information retrieval metrics.

---

## Features

- Custom anime screenshot dataset
- Automatic frame extraction from anime episodes
- Four embedding extraction methods
- Cosine Similarity retrieval
- Retrieval evaluation using multiple metrics
- Modular project structure

---

## Embedding Models

The following embedding extraction methods were implemented and compared:

- Gemini Embedding 2 (Original)
- Gemini Embedding 2 (512×512)
- ResNet50 (Classification)
- ResNet50 (Triplet Loss)

---

## Retrieval Pipeline

```text
Input Screenshot
        │
        ▼
Embedding Extraction
        │
        ▼
Cosine Similarity Search
        │
        ▼
Top-5 Similar Images
        │
        ▼
Predicted Anime
```

---

## Dataset

The dataset was built from anime episodes using an automatic frame extraction pipeline.

Final dataset:

- 23 anime titles
- 24,009 images

Dataset split:

- Train — 70%
- Validation — 15%
- Test — 15%

To ensure fair evaluation, the retrieval database used during testing contains **only embeddings generated from the training subset**. Test images are used exclusively as search queries.

---

## Evaluation Metrics

The following retrieval metrics are used:

- Recall@1
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (mAP)

Evaluation results are automatically exported to:

```
results/evaluation_metrics.json
results/evaluation_metrics.csv
```

---

## Project Structure

```text
anime_project/

├── config/
├── data/
├── docs/
├── embeddings/
├── logs/
├── models/
├── notebooks/
├── results/
├── src/
├── tests/

├── README.md
├── requirements.txt
└── .env.example
```

---

## Installation

Clone the repository:

```bash
git clone <repository_url>
cd anime_project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your Gemini API key.

---

## Results

The project compares four embedding extraction methods and evaluates their retrieval quality using standard Image Retrieval metrics.

Final evaluation results are available in:

- `results/evaluation_metrics.csv`
- `results/evaluation_metrics.json`

---

## Documentation

Additional project documentation is available in:

```
docs/technical_report.md
```

---

## Future Work

Possible improvements include:

- Larger dataset
- Additional embedding models
- FAISS integration
- Faster nearest neighbor search
- Larger anime collection

---

## License

This project was developed for educational and research purposes.

# Usage

The project is organized as a collection of Jupyter notebooks, where each notebook represents a separate stage of the Image Retrieval pipeline.

Run the notebooks in the following order:

### 1. Dataset Preparation

Run:

```
preprocessing.ipynb
```

This notebook:

- prepares the dataset;
- performs preprocessing;
- splits the dataset into training, validation, and test subsets.

---

### 2. Train the Models

Train the ResNet50 classifier:

```
training_resnet50.ipynb
```

Train the ResNet50 model using Triplet Loss:

```
training_resnet50_triplet.ipynb
```

Both notebooks save the best trained models to the `models/` directory.

---

### 3. Generate Embeddings

Generate embedding databases for each approach:

```
Gemini_Original_embeddings.ipynb
Gemini_512_embeddings.ipynb
ResNET50_embeddings.ipynb
ResNET50_triplet_embeddings.ipynb
```

Each notebook creates the corresponding embedding database used for retrieval.

---

### 4. Evaluate the Models

Run:

```
evaluation.ipynb
```

The evaluation is performed using:

- training embeddings as the retrieval database;
- test images as search queries.

This evaluation protocol prevents data leakage and provides an unbiased comparison between models.

The following metrics are computed:

- Recall@1
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (mAP)

Evaluation results are automatically saved to:

```
results/evaluation_metrics.json
results/evaluation_metrics.csv
```

---

### 5. Search for Anime

Run:

```
similarity_search.ipynb
```

The search pipeline performs the following steps:

1. Upload an anime screenshot.
2. Select one of the available embedding models.
3. Generate an embedding for the uploaded image.
4. Compare it with the embedding database using cosine similarity.
5. Display the Top-5 most similar images together with the predicted anime title.

---

## Workflow

```
Dataset Preparation
        │
        ▼
Model Training
        │
        ▼
Embedding Generation
        │
        ▼
Model Evaluation
        │
        ▼
Similarity Search
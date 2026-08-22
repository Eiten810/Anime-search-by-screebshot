# Technical Report

## Anime Search by Screenshot

### Project Overview

The goal of this project is to develop an image retrieval system capable of identifying an anime title from a user-uploaded screenshot.

Unlike image classification, the system does not directly predict a class. Instead, it extracts a feature representation (embedding) of the query image and searches for the most similar images in a database using cosine similarity.

The project compares several embedding extraction approaches to determine which model provides the best retrieval quality.

---

# Dataset

A custom dataset was created specifically for this project.

Initially, the dataset contained many low-quality and irrelevant images. After manual cleaning, it became clear that the remaining number of images was insufficient for training and evaluation.

To solve this problem, a preprocessing script was developed to automatically extract frames directly from anime episodes, allowing the dataset to be rebuilt from high-quality screenshots.

The final dataset contains **24,009 images** belonging to **23 anime titles**.

The dataset was divided into three subsets:

- Train — 70%
- Validation — 15%
- Test — 15%

To prevent data leakage, the test subset was never used during model training or embedding database construction.

---

# Embedding Extraction

Four different approaches were evaluated.

## Gemini Embedding 2 (Original)

Images are processed in their original resolution without resizing before embeddings are extracted using Gemini Embedding 2.

---

## Gemini Embedding 2 (512×512)

Images are resized to **512×512** before embedding extraction in order to evaluate the influence of image resolution.

---

## ResNet50 Classification

A ResNet50 classifier was trained on the training subset.

After training, the classification head was removed and the feature extractor was used to generate image embeddings.

Input image size:

- 224×224

---

## ResNet50 Triplet Loss

To improve retrieval quality, an additional model based on Triplet Loss was implemented.

Instead of learning class probabilities, the network learns an embedding space where images from the same anime are located closer together while images from different anime are pushed farther apart.

---

# Retrieval Pipeline

The search process consists of the following steps:

1. Upload an image.
2. Generate its embedding.
3. Compare the embedding with the database using cosine similarity.
4. Sort images by similarity score.
5. Return the Top-5 most similar results.

---

# Evaluation Protocol

To ensure a fair comparison between embedding methods, retrieval evaluation was performed using only unseen images.

The retrieval database was built **only from training embeddings**.

Each query image originated from the **test subset**.

Therefore, no test image was present inside the search database, eliminating information leakage and providing an objective evaluation.

---

# Evaluation Metrics

The following retrieval metrics were used:

- Recall@1
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (mAP)

The evaluation results were automatically exported to:

- `results/evaluation_metrics.json`
- `results/evaluation_metrics.csv`

---

# Development Challenges

Several important problems were solved during development.

### Dataset Quality

The initial dataset contained many noisy images.

A preprocessing pipeline was developed to automatically extract screenshots from anime episodes, resulting in a significantly larger and cleaner dataset.

### Image Resolution

Originally, every image was resized to **224×224**.

Later, separate preprocessing strategies were introduced:

- Gemini Original — original resolution
- Gemini 512 — 512×512
- ResNet50 — 224×224

This allowed a fair comparison between different embedding models.

### Retrieval Model

The initial ResNet50 classifier produced satisfactory classification results but was not specifically optimized for image retrieval.

Therefore, an additional embedding model based on Triplet Loss was implemented and compared with the classification-based approach.

---

# Results

The project compares four embedding extraction methods:

- Gemini Embedding 2 (Original)
- Gemini Embedding 2 (512×512)
- ResNet50 Classification
- ResNet50 Triplet Loss

The obtained retrieval metrics make it possible to compare both retrieval quality and computational efficiency of different embedding methods.

---

# Future Work

The proposed architecture is modular and can be extended in several directions:

- increase the dataset size;
- add new embedding models;
- integrate approximate nearest neighbor search (FAISS);
- support larger anime collections.

---

# Conclusion

A complete anime retrieval system based on image embeddings was successfully developed.

The project includes:

- custom dataset generation;
- model training and comparison;
- embedding extraction;
- cosine similarity search;
- quantitative retrieval evaluation.

The modular design makes the system suitable for future improvements and experiments with more advanced retrieval models.



	                    Recall@1	Recall@5	Recall@10	MRR	        mAP	        Avg Time (s)
Gemini Original	        0.899500	0.966685	0.984731	0.928905	0.883375	1.044567
Gemini 512	0.911993	0.968073	0.983343	0.936182	0.901549	0.903713    0.903713
ResNet50 Classification	0.640478	0.816768	0.890894	0.717141	0.622366	0.036621
ResNet50 Triplet	    0.817046	0.916991	0.944475	0.859901	0.831519	0.018288
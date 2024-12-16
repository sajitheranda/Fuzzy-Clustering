# Fuzzy Clustering

## In-Class Project 1 - MA4144

This project focuses on implementing fuzzy clustering algorithms to analyze a dataset of social media posts. The dataset includes features related to post content and engagement (e.g., likes and presence of specific words). The project involves implementing clustering techniques and analyzing the results.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Tasks Overview](#tasks-overview)
   - [1: Data Loading and Preparation](#1-data-loading-and-preparation)
   - [2: Initialize Membership Matrix](#2-initialize-membership-matrix)
   - [3: Compute Cluster Centers](#3-compute-cluster-centers)
   - [4: Update Membership Matrix](#4-update-membership-matrix)
   - [5: Implement Fuzzy Clustering Algorithm](#5-implement-fuzzy-clustering-algorithm)
   - [6: Analyze Hyperparameters](#6-analyze-hyperparameters)
   - [7: Evaluate with Silhouette Score](#7-evaluate-with-silhouette-score)
   - [8: Visualize Clusters](#8-visualize-clusters)
   - [9: Compare Clusters to Raw Data](#9-compare-clusters-to-raw-data)
4. [Dataset Description](#dataset-description)
5. [How to Run](#how-to-run)
6. [References](#references)

---

## Introduction

This project applies fuzzy clustering to analyze a dataset of social media posts. The primary goal is to cluster posts based on their textual content and number of likes. The implementation involves writing custom Python functions for:
- Membership matrix initialization
- Cluster center calculations
- Membership updates

The project also explores hyperparameter tuning and cluster evaluation metrics, such as Silhouette Score.

---

## Setup

To get started, import the necessary libraries:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
Make sure the required files (`SMDataRefined.csv` and `SMData.csv`) are present in the project directory.

---

## Tasks Overview

### 1: Data Loading and Preparation
- **Goal:** Load the dataset (`SMDataRefined.csv`) into a Pandas DataFrame, then convert it into a NumPy matrix (`X`).
- **Steps:**
  1. Display the data.
  2. Extract features into a matrix (`X`).
  3. Define `N` (number of rows) and `M` (number of columns).
- **Dataset Info:**
  - The first column contains standardized `likes`.
  - Subsequent columns indicate the presence (`1`) or absence (`0`) of specific tokens in the post.

### 2: Initialize Membership Matrix
- **Goal:** Create a function `initMu` to initialize a membership matrix `U` for `N` points and `c` clusters.
- **Formula:**
  - `0 ≤ u_ij ≤ 1`
  - `Σ u_ij = 1` for each `i`
- **Steps:**
  1. Implement random initialization.
  2. Ensure constraints are met.

### 3: Compute Cluster Centers
- **Goal:** Write a function `calculateCenters` to compute cluster centers (`v_j`).
- **Formula:**
  ```
  v_j = Σ (u_ij^m1 * x_i) / Σ (u_ij^m1)
  ```
  - `u_max` is the maximum membership value.
  - `m1` is a hyperparameter.
- **Steps:**
  1. Use matrix multiplications to efficiently calculate centers.

### 4: Update Membership Matrix
- **Goal:** Implement `updateMu` to update the membership matrix `U`.
- **Formula:**
  ```
  u_ij = [Σ (d_ij / d_il)^(2/(m2-1))]^-1
  ```
  - `d_ij` is the distance between the `i`-th data point and the `j`-th cluster center.
  - `m2` is a hyperparameter.
- **Steps:**
  1. Use `sklearn.metrics.pairwise_distances` to compute distances.
  2. Avoid for-loops and utilize NumPy for vectorized operations.

### 5: Implement Fuzzy Clustering Algorithm
- **Goal:** Create a function `fuzzyClustering` to cluster data.
- **Steps:**
  1. Initialize membership matrix `U`.
  2. Repeat the following until convergence or max iterations are reached:
     - Compute cluster centers.
     - Update membership matrix `U`.
     - Calculate norm difference between consecutive `U` matrices.
  3. Plot `U` norm difference vs iterations.

### 6: Analyze Hyperparameters
- **Goal:** Run `fuzzyClustering` with different values of `m1`, `m2`, and `nclusters`.
- **Steps:**
  1. Observe and plot the variation in `U` norm differences.
  2. Assign crisp cluster labels using `numpy.argmax` on `U`.

### 7: Evaluate with Silhouette Score
- **Goal:** Use `sklearn.metrics.silhouette_score` to determine the best hyperparameters (`m1`, `m2`, `nclusters`).
- **Steps:**
  1. Perform grid search over:
     - `m1 ∈ {1.0, 1.2, ..., 3.0}`
     - `m2 ∈ {1.2, 1.4, ..., 3.0}`
     - `nclusters ∈ {2, 3, ..., 15}`
  2. Report the best combination based on the Silhouette Score.

### 8: Visualize Clusters
- **Goal:** Create visualizations of clusters in 2D.
- **Steps:**
  1. Implement `visualizeClusters2D`.
  2. Plot data points in different clusters with colors for specific feature pairs (`f1`, `f2`).

### 9: Compare Clusters to Raw Data
- **Goal:** Match the clustering results to the original dataset (`SMData.csv`).
- **Steps:**
  1. Load the raw dataset.
  2. Match entries with cluster labels (`yfuzzy`).
  3. Observe patterns and evaluate clustering success.

---

## Dataset Description

1. **`SMDataRefined.csv`**:
   - Processed dataset containing the number of likes and token presence.
   - Columns:
     - `likes`: Standardized engagement metric.
     - `TextToken_*`: Binary indicators for token presence.

2. **`SMData.csv`**:
   - Original dataset with raw text and features.
   - Columns:
     - `Text`: Social media post content.
     - `likes`: Engagement metric.

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/sajitheranda/Fuzzy-Clustering.git
   cd Fuzzy-Clustering
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook
   ```
4. Complete the tasks in order, and export the notebook as a PDF:
   - File > Save and Export Notebook As > 
---

## References

- [Fuzzy Clustering - Wikipedia](https://en.wikipedia.org/wiki/Fuzzy_clustering)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

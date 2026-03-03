# SRDM
**SRDM: Self-Reward Diffusion Models for Image Reasoning Generation**

SRDM is a research framework for improving **multi-dimensional reasoning ability** in diffusion models via **self-reward reinforcement learning**, built upon **DDPO (Denoising Diffusion Policy Optimization)**.

This repository redesigns the reward mechanism to reduce reliance on external reward models (e.g., CLIP/VLMs/RLHF) and targets reasoning-oriented metrics such as **counting accuracy**, **spatial consistency**, and **transform invariance**.

---

## 1. Motivation

Most RL + Diffusion pipelines depend on **external reward models**, such as:

- Human feedback reward models (RLHF)
- Vision-language models (CLIP, GPT-4V, Qwen-VL)
- External scoring pipelines

However, for image reasoning generation, external rewards often suffer from:

1. **High cost** (annotations / API usage)
2. **Semantic bias** without reliable **structural reasoning**
3. **Bias amplification** during PPO-style optimization, harming stability
4. **No supervision for early diffusion steps** (only end images are scorable)

Even when aesthetics improve, diffusion models can still be weak in:

- Relative position reasoning
- Counting accuracy
- Spatial consistency
- Transform invariance

SRDM redesigns rewards to optimize **reasoning structure**, not just semantic preference.

---

## 2. Core Ideas & Contributions

### 2.1 Intrinsic Reward for Diffusion (`r_in`)
We construct an **intrinsic reward** from the diffusion model itself.

Inspired by maximum-entropy viewpoints and TTRL-style relative preference learning:

- For a fixed prompt, higher-probability generations correspond to higher implicit reward.
- We approximate intrinsic reward from diffusion trajectory statistics (e.g., log-prob / policy log-likelihood).

Formally:
$r_{\text{in}} \propto \log p_\theta(x_0 \mid c)$

This stage aims to validate:
- **No external supervision**
- Reward available **internally** and potentially **along diffusion steps**
- A clean baseline before adding structure-aware rewards

---

### 2.2 Structure-Aware Clustering Consistency Reward (`r_cluster`)
Intrinsic reward alone does not necessarily improve *reasoning structure*.

We introduce a **clustering consistency reward**:

- For each prompt, sample multiple images (multi-trajectory generation)
- Extract **structure features**
- Cluster samples in feature space
- Use **majority voting** / density thresholds to assign rewards

Intuition:
> Frequent structural modes under the model distribution represent more stable structure patterns.

---

### 2.3 Optional Inference Consistency Module (`ICM`)
We optionally train a lightweight module:

- Input: image (or structured representation)
- Output: reasoning consistency score in $[0,1]$
- Supervised by self-reward signals (e.g., clustering reward)
- Used to stabilize or correct noisy clustering signals


---

## 3. Structural Signal: Ground DINO as an Observer

Instead of token-grid clustering (VQ) or classical CV heuristics, SRDM uses **Ground DINO** to extract object-level structure:

- Use **spaCy** to extract nouns, quantity terms, and spatial relation triplets from prompt
- Feed object names into Ground DINO
- Obtain bounding boxes, centroids, instance counts
- Convert outputs into objective **structure feature vectors**

We experimentally found that the following are not reliable for this task:
- VQ-Encoder token clustering
- OpenCV heuristics
- YOLO / SAM / dense segmentation baselines

Ground DINO provides more stable object-level structure signals for reasoning prompts.

---

## 4. Feature Engineering (Reasoning Dimensions)

Given Ground DINO detections (boxes + centroids), we design features for:

### 4.1 Counting Feature
- Object count per queried class

### 4.2 Distance Feature
Pairwise centroid distance statistics:
- min / max / mean

### 4.3 Overlap Feature
- Box intersection / overlap ratio statistics

### 4.4 Ordering Feature (Discrete)
- Encode ordering patterns (x/y sorting) and compare via **Hamming-like** distances

### 4.5 Scale-Class Feature
- Average box area per class (vector over classes)

### 4.6 Relative Position Feature (Discrete)
For each relation triplet $(A, rel, B)$:
- Compute mean centroid of A and B
- Encode $(\Delta x, \Delta y)$ into $\{-1,0,1\}$ with thresholds

---

## 5. Clustering & Reward Construction

Different features use different clustering strategies:

- **Counting**: natural-number majority voting (no clustering needed)
- **Continuous features** (distance/overlap): KDE / DBSCAN / density-based clustering
- **Discrete features** (ordering/position): k-modes or majority voting over patterns
- **High-dimensional** (scale-class): cosine similarity clustering (optional sparsification)

Reward assignment follows majority voting / thresholding:
- If a cluster size exceeds a threshold (e.g., $\alpha \cdot \frac{N}{M}$), samples in that cluster receive reward 1; otherwise 0.
- If structure is non-discriminative (all 0 or all 1 after normalization), reward is nullified.

---

## 6. Training Pipeline

Sampling mode:
- **Single prompt, multi-trajectory generation**
- Cluster at the diffusion end ($x_0$), then assign rewards

### Stage 1: Intrinsic Reward Training (Stage-4 in roadmap)
- Replace DDPO external reward with intrinsic reward `r_in`
- Run full PPO/DDPO training loop

### Stage 2: Add Clustering Consistency Reward (Stage-5)
- Compute `r_cluster` using Ground DINO + feature clustering
- Train with combined reward

### Stage 3 (Optional): Add ICM (Stage-6)
- Train ICM with self-reward supervision
- Joint reward training

---

## 7. Dataset & Evaluation

### Training prompts
- **FLUX-Reason-6M** (prompt subset only)

We use category labels to sample reasoning-oriented prompts.
We do **not** use images or chain-of-thought as supervision.

### Evaluation
- **PRISM-Bench** official evaluation pipeline

Metrics:
- Spatial consistency
- Transform invariance
- Counting accuracy

External VLMs may be used **only for benchmarking**, not training.

---

## 8. Research Roadmap (Aligned with Outline v2)

1. Build DDPO + SDXL baseline
2. Reproduce CLIP-based DDPO training loop
3. Select a structural observer tool (Ground DINO)
4. **Stage-4: intrinsic reward `r_in` (pure self-reward baseline)**
5. Stage-5: add clustering consistency reward `r_cluster`
6. Stage-6: optional ICM module
7. Stage-7: benchmark on PRISM-Bench
8. Stage-8: paper writing & ablations

---

## 9. Acknowledgement

This project is built upon the official **DDPO** implementation.

All diffusion sampling and PPO optimization follow the original DDPO pipeline, while SRDM redesigns the reward mechanism for self-supervised reasoning enhancement.
# Compositional Concept Synthesis via CLIP-Guided Primitive Stitching

Welcome! 👋 This project tackles the challenge of generating complex visual concepts by intelligently stitching together simple visual primitives.  
Given a set of basic shapes (e.g., "person", "umbrella"), we aim to arrange them into a scene (e.g., "a person under an umbrella") such that the final rasterized image matches the concept described in natural language.  
We use the power of **CLIP** to compare image and text embeddings via **cosine similarity**. 

---

## 🚀 What This Project Does

- Takes a list of primitive visual elements and stitches them together into a composed scene
- Optimizes layout using:
  - **PGPE** (Policy Gradients with Parameter-based Exploration)
  - **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy)
  - **Gradient-based optimization**
- Analyzes the loss landscape and proposes ways to smooth it for better convergence

---

## 🎓 Academic Context

This project was developed as part of the 🎓 **"Deep Learning & Applied AI"** course at **Sapienza University of Rome**, academic year **2023/2024**.

---

## 📄 Full Report

For all the details we provide the full report here:  
👉 [📘 View Report](https://github.com/pietrosig/dl_project/blob/main/report.pdf)

---

## 🧑‍💻 Authors

This project was created by:

- **Paolo Cursi**  
- **Pietro Signorino**

# 🗣️ Native Language Identification of Indian English Speakers Using HuBERT  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

---

## 🎯 Objective  

This project aims to identify the **native language (L1)** of Indian speakers while they speak English — just by listening to their **accent patterns**.  

In India, English sounds different depending on where a person is from — a Tamil speaker’s English doesn’t sound quite like a Punjabi’s, right?  
The goal here was to build a model that can *detect those subtle accent cues* and predict the speaker’s native language.  

We explored both:
- 🎵 **Traditional features** like MFCCs (handcrafted acoustic features), and  
- 🧠 **HuBERT embeddings**, which are self-supervised representations learned from raw speech.  

The project also tested **how well the model generalizes** across:
- 👩‍🦰 Adults vs 👦 Children  
- 🗣️ Word-level vs 🗯️ Sentence-level speech  

Finally, we used the model in a fun, real-world demo — a **restaurant app** that suggests cuisines based on a customer’s accent!

---

## ⚙️ Project Scope  

### 🔹 1. Native Language Identification Model
- Built a model that predicts a speaker’s **L1** (like Hindi, Tamil, Telugu, Malayalam, or Kannada).  
- Compared two approaches:
  - **MFCCs** → traditional handcrafted features.  
  - **HuBERT embeddings** → deep, contextual speech features.  
- Conducted a **layer-by-layer HuBERT analysis** to see which layer captures accent cues best.  
- Tried different models — **CNN**, **BiLSTM**, and **Transformer-based** architectures.  
- Tuned hyperparameters for better accuracy and generalization.

---

### 🔹 2. Age-Based Generalization
We trained the model using **adult voices** and tested it on **children’s voices** to see how age affects accent patterns.  
This helped analyze which features (MFCC vs HuBERT) are more stable across age groups.

---

### 🔹 3. Word-Level vs Sentence-Level
We compared performance on both **word-level** and **sentence-level** clips to check which gives more consistent accent cues.  
Turns out, longer sentences often carry richer prosodic and rhythmic information.

---

### 🔹 4. Accent-Aware Cuisine Recommendation 🍽️
To make things more interesting, we built an application that uses accent detection to recommend **regional cuisines**.  

Here’s how it works:
1. A customer speaks a short English phrase (like “Can I get a coffee?”).  
2. The system detects their **accent** — for example, Malayalam-English or Punjabi-English.  
3. Based on that, it infers the **region** and suggests food from that culture.  

| Detected Accent | Region | Suggested Dishes |
|------------------|---------|-----------------|
| Malayalam-English | Kerala | Appam, Puttu, Avial |
| Punjabi-English | Punjab | Butter Chicken, Amritsari Kulcha |
| Tamil-English | Tamil Nadu | Dosa, Pongal, Sambar |

This simple idea shows how **speech analytics** can make digital interactions more personal and culturally aware.

---

## 🗂️ Dataset  

The model was trained using the **IndicAccentDb** dataset available on Hugging Face:  
📎 [IndicAccentDb – Hugging Face](https://huggingface.co/datasets/DarshanaS/IndicAccentDb)

It includes recordings of Indian speakers from various native language backgrounds, in both **adult** and **child** categories, and at **word** and **sentence** levels.

---

## 🧠 Conceptual Background  

An **accent** reflects the phonetic influence of a speaker’s **first language (L1)**.  
When Indian speakers speak English, they unconsciously carry over sound patterns from their mother tongue — like how vowels are pronounced or how syllables are stressed.  

This project uses **HuBERT (Hidden-Unit BERT)**, a **self-supervised speech model**, to explore how such deep representations encode accent cues and how these cues generalize across speakers, ages, and contexts.

---

## 🧩 Tech Stack  

| Category | Tools & Frameworks |
|-----------|--------------------|
| Language | Python |
| ML Framework | PyTorch, Transformers |
| Feature Extraction | Librosa (MFCCs), HuBERT |
| Visualization | Matplotlib |
| Web App | Streamlit |
| Data Handling | NumPy, Pandas |


📊 Results & Insights
Experiment	Feature	Model	Accuracy	Observation
Adult Speech	MFCC	CNN	99%	Baseline accuracy
Adult Speech	HuBERT	BiLSTM	98%	Better with contextual features
Cross-Age	HuBERT	Transformer	99%	Good generalization
Sentence-Level	HuBERT	Transformer	99%	Richer speech context helps




👩‍💻 Project Lead: **Sharon Elsa Binu, Joel Garvaziz, Meera Krishna S**


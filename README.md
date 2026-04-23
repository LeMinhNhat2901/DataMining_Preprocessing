# Data Mining and Applications - Assignment 1: Data Preprocessing

**CSC14004 - Khai thác dữ liệu và ứng dụng**  
University of Science, VNU-HCM (HCMUS) — Semester 2, 2025/2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Team Members](#2-team-members)
3. [Datasets](#3-datasets)
4. [Project Structure](#4-project-structure)
5. [Technical Approach](#5-technical-approach)
6. [How to Run](#6-how-to-run)
7. [Dependencies](#7-dependencies)
8. [Academic Information](#8-academic-information)

---

## 1. Project Overview

This repository contains the full implementation for Assignment 1 of the course *Data Mining and Applications (CSC14004)*. The assignment focuses on building a comprehensive data preprocessing pipeline across three distinct data modalities: image data, tabular data, and text data.

Each modality is treated with rigorous statistical analysis before any transformations are applied, following the principle that preprocessing decisions should be grounded in empirical evidence rather than convention. For each technique implemented, the group conducted ablation studies and quantitative evaluations to measure and justify the impact of every preprocessing step.

The scope of the assignment covers:

- **Part 1 — Image Data:** Statistical analysis and preprocessing of the Intel Image Classification dataset (approx. 14,000 natural scene images across 6 classes). Tasks include pixel distribution analysis, duplicate detection via perceptual hashing, resize quality evaluation (SSIM/PSNR), color space comparison with PCA, normalization strategies validated with KS tests, data augmentation evaluated with t-SNE, and advanced techniques such as edge detection (Sobel, Prewitt, Canny) with ANOVA testing.

- **Part 2 — Tabular Data:** End-to-end preprocessing of the Rain in Australia dataset (145,460 rows, 23 columns). Tasks include normality testing (D'Agostino-Pearson), multicollinearity analysis, missing data mechanism classification (Little's MCAR test, Chi-square), imputation comparison (Mean/Median/Mode/k-NN/MICE), outlier detection (IQR, Z-score, Isolation Forest, LOF, DBSCAN), robust scaling, categorical encoding (Binary, Target, Frequency), feature selection (ANOVA, Mutual Information, Random Forest, RFE, PCA, UMAP), and class imbalance handling (SMOTE, ADASYN).

- **Part 3 — Text Data:** Preprocessing pipeline for the IMDB 50K Movie Reviews dataset. Tasks include text length distribution analysis (Mann-Whitney U test), word cloud and TTR analysis, Zipf's Law verification, a multi-step normalization pipeline (lowercasing, HTML removal, tokenization), stop-word impact analysis (MI and Naive Bayes), stemming/lemmatization comparison (collision rate and Logistic Regression CV), and text vectorization (BoW, TF-IDF n-gram, Word2Vec, Sentence Transformer).

- **Link dataset**
  - **Part 1:** https://www.kaggle.com/datasets/puneet3806/intel-image-classification
  - **Part 2:** https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
  - **Part 3:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

- **Link Google Drive:** https://drive.google.com/drive/folders/1v2XAsuN0Nd4FLp8O9LrvzkcSq5-SQRo1?usp=sharing

---

## 2. Team Members

**Group 8**

| No. | Student ID | Full Name | Responsibilities | Completion |
| :-: | :---: | :--- | :--- | :---: |
| 1 | 23120067 | Le Minh Nhat | Part 1 — Statistical analysis, image preprocessing techniques, report writing, make readme.md, check final version, aggregate & finalize repo, documentation | 100% |
| 2 | 23120062 | Tran Kim Ngoc | Part 1 — Preprocessing techniques, impact analysis, advanced image processing | 100% |
| 3 | 23120047 | Nguyen Gia Huy | Part 2 — Normalization, encoding, feature selection, class imbalance handling | 100% |
| 4 | 23120063 | Nguyen Thanh Nguyen | Part 2 — EDA, missing data handling, outlier detection | 100% |
| 5 | 23120038 | Le Hoang My Ha | Part 3 — Text data preprocessing pipeline | 100% |

---

## 3. Datasets

### Part 1: Intel Image Classification

- **Source:** [Kaggle — Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Description:** A multi-class natural scene classification dataset originally released by Intel. Contains approximately 25,000 images at 150×150 pixels across 6 categories: *buildings, forest, glacier, mountain, sea, street*.
- **Why this dataset:** Exceeds the minimum requirement of 5,000 images and 5 classes. The imbalance ratio (IR ≈ 1.15) confirmed the dataset is well-balanced, making it suitable for evaluating preprocessing techniques in isolation from class imbalance effects.

### Part 2: Rain in Australia

- **Source:** [Kaggle — Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
- **File:** `weatherAUS.csv`
- **Description:** Real weather observations collected over multiple years at weather stations across Australia. Contains 145,460 rows and 23 columns, combining 16 numerical attributes, 6 categorical attributes, and 1 datetime column. The target variable is *RainTomorrow* (binary: Yes/No).
- **Why this dataset:** Satisfies all requirements — large scale, mixed feature types, and a structural missing data rate of 10.73% (with some columns exceeding 40% missing, e.g., *Sunshine* at 48.0%).

### Part 3: IMDB 50K Movie Reviews

- **Source:** [Kaggle — IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Description:** A standard NLP benchmark dataset for sentiment analysis. Contains 50,000 English-language movie reviews labeled as *positive* or *negative*, balanced at 25,000 samples per class (IR = 1.00).
- **Why this dataset:** Exceeds the minimum of 10,000 samples with 2 classification labels. The raw text contains HTML tags, special characters, and informal language, making it a realistic testbed for a complete NLP preprocessing pipeline.

---

## 4. Project Structure

```
DataMining_Preprocessing/
│
├── data/                        # Raw and processed datasets (not tracked by git, using link Google Drive that link to this github)
│
├── docs/                        # Supporting documentation
│   ├── report.pdf               # Full report in PDF format
│   ├── CSC14004 - Data Mining - P1.pdf  # Assignment specification
├── notebooks/                   # Jupyter notebooks (main deliverables)
│   ├── 01_EDA_image.ipynb           # Part 1: Image data statistical analysis
│   ├── 02_preprocessing_image.ipynb # Part 1: Image preprocessing pipeline
│   ├── 03_EDA_tabular.ipynb         # Part 2: Tabular data EDA
│   ├── 04_preprocessing_tabular.ipynb  # Part 2: Tabular preprocessing pipeline
│   ├── 05_EDA_text.ipynb.ipynb      # Part 3: Text data EDA
│   └── 06_preprocessing_text.ipynb  # Part 3: Text preprocessing pipeline
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project summary, problem statement, datasets, structure and team members
```

---

## 5. Technical Approach

### Part 1: Image Data

The preprocessing workflow was evaluated using k-NN (k=5) on flattened 64×64 pixel vectors as a non-parametric baseline — chosen because any improvement in preprocessing quality directly translates into tighter cluster separation in Euclidean space, which k-NN can exploit without parameterization bias.

Key findings and decisions:
- **Resize:** 128×128 was selected (SSIM = 0.858, PSNR = 25.60 dB), accepting a minor k-NN accuracy drop in exchange for feature fidelity that benefits deep learning architectures downstream.
- **Color Space:** HSV outperformed RGB, Grayscale, and LAB on k-NN accuracy (0.4453), because separating the Value channel reduces illumination sensitivity — critical for outdoor scene classification.
- **Duplicate Detection:** 188 images were removed (1.34% redundancy rate) using pHash with BallTree indexing, preventing data leakage between train/test splits.
- **Class Balance:** IR ≈ 1.15 confirmed no intervention was required.

### Part 2: Tabular Data

- **Missing Data:** Little's MCAR test (p ≈ 0.000) and Chi-square independence testing identified the missing mechanism as MAR (Missing At Random), correlated with the *Location* attribute (Cramér's V up to 0.9094 for some features). MICE (IterativeImputer) was selected as the imputation strategy, achieving the best RMSE ≈ 4.97 in a controlled ground-truth experiment with 10% artificially masked data.
- **Outliers:** KS test confirmed that dropping outliers would distort the distributions of all 16 numeric columns. Since extreme weather events (heavy rain, strong winds) carry high predictive signal for the *RainTomorrow* label, outliers were preserved and RobustScaler was applied instead.
- **Encoding:** Binary Encoding (6 columns for 49 location categories) and Target Encoding with 5-fold smoothing were applied. VIF checks confirmed no new multicollinearity was introduced (max VIF = 1.73).
- **Class Imbalance:** IR = 0.28 (No: 78.1% / Yes: 21.9%). SMOTE actually reduced F1 from 0.7489 to 0.7381 due to synthetic noise at the decision boundary. The final decision was to use `class_weight='balanced'` instead of resampling.

### Part 3: Text Data

- **Normalization Pipeline:** Reduced vocabulary size from 390,931 to 232,098 tokens (-40.6%), with the special character removal step contributing the largest reduction (-36.43% of vocabulary).
- **Tokenization:** Subword BPE was identified as optimal — vocabulary size of 7,879 with OOV rate of 0.004%, balancing coverage and dimensionality.
- **Stop Word Removal:** Increased Naive Bayes test accuracy from 84.36% to 85.72%, despite a marginal decrease in average Mutual Information (the removed stop words had near-zero MI).
- **Stemming/Lemmatization:** Baseline (no normalization) achieved the highest Logistic Regression CV accuracy (84.90%), suggesting that preserving inflectional variants retains sentiment-relevant information in this domain.
- **Vectorization:** TF-IDF (1-3 gram, 5,000 features) achieved 77.2% SVM accuracy vs. 74.4% for Sentence Transformer (`all-MiniLM-L6-v2`), though the Transformer showed higher silhouette score (0.0254 vs. 0.0240), indicating better semantic clustering. The gap in classification performance is attributed to the off-the-shelf usage without domain-specific fine-tuning.

---

## 6. How to Run

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Step 1: Clone the repository

```bash
git clone https://github.com/LeMinhNhat2901/DataMining_Preprocessing.git
cd DataMining_Preprocessing
```

### Step 2: Create a virtual environment (recommended)

Using `venv`:
```bash
python -m venv venv python=3.10

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Using `conda`:
```bash
conda create -n dm_preprocessing python=3.10
conda activate dm_preprocessing
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK and spaCy assets

Some notebooks require additional runtime assets that cannot be distributed via pip:

```bash
python -m nltk.downloader averaged_perceptron_tagger_eng wordnet omw-1.4
python -m spacy download en_core_web_sm
```

### Step 5: Download the datasets

Download the datasets from Kaggle and place them under the `data/` directory:

- Intel Image Classification: extract to `data/intel_image/`
- Rain in Australia (`weatherAUS.csv`): place in `data/tabular/`
- IMDB 50K Movie Reviews (`IMDB Dataset.csv`): place in `data/text/`

### Step 6: Run the notebooks

Open Jupyter and run the notebooks in the `notebooks/` folder in sequential order (01 through 06). Each notebook is self-contained and documents the theory, implementation, and results for its respective section.

```bash
jupyter notebook
```

---

## 7. Dependencies

| Category | Libraries |
| :--- | :--- |
| Core | `numpy`, `pandas` |
| Visualization | `matplotlib`, `seaborn` |
| Statistical Testing | `scipy`, `statsmodels` |
| Machine Learning | `scikit-learn` |
| Image Processing | `opencv-python`, `Pillow`, `scikit-image`, `imagehash` |
| NLP | `nltk`, `spacy`, `tokenizers`, `gensim`, `sentence-transformers` |
| Missing Data | `missingno` |
| Utilities | `tqdm`, `wordcloud`, `jupyter`, `ipykernel` |

Full version-pinned list: see `requirements.txt`.

---

## 8. Academic Information

### Course Details

- **Course:** CSC14004 — Data Mining and Applications (Khai thác dữ liệu và ứng dụng)
- **Assignment:** Assignment 1 — Data Preprocessing
- **Semester:** HK2, Academic Year 2025/2026
- **Faculty:** Faculty of Information Technology
- **University:** University of Science, Vietnam National University Ho Chi Minh City (HCMUS)

### Instructors

- **Nguyen Thi Thu Hang** - <ntthang@fit.hcmus.edu.vn> 
- **Nguyen Ngoc Duc** - <nnduc@fit.hcmus.edu.vn>
- **Le Nhut Nam** - <lnnam@fit.hcmus.edu.vn>

### Academic Integrity

All code in this repository is the original work of the group members listed above. External libraries are used in accordance with their respective licenses and are cited where applicable in the notebooks. No unauthorized collaboration or code sharing has taken place.

### Support and Questions

For course-related questions or technical support:

*   **Primary Channel:** Use the course's ZALO group.
*   **Office Hours:** As announced by the instructors.
*   **Project Issues:** Contact the project lead via email or GitHub.

---

**© 2026 University of Science (VNU-HCMC)**  
*Developed for Data Mining and Applications*

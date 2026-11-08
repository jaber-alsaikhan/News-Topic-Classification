# Headline-only News Topic Classification

**NLP course project, Option 3 (News Topic Classification), Section 1, Group 2.** Imam Abdulrahman Bin Faisal University, College of Computer Science and Information Technology.

Given **one English news headline**, the notebook predicts its topic: **World, Sports, Business or Sci/Tech**. It compares a traditional method with a pretrained one on the AG News headlines, focusing on the hardest pair of classes, Business vs Sci/Tech.

| Role | Name | Student ID |
|---|---|---|
| Team leader | Mohammed Saeed Alamodi | 2240009007 |
| Member 2 | Abdulaziz Mohammed Alkhlawi | 2240006095 |
| Member 3 | Jaber Saleh Alsaikhan | 2240006539 |

**Experimental question.** *If only the headline is available, does a pretrained sentence-embedding classifier separate Business and Sci/Tech headlines more reliably than TF-IDF logistic regression?*

## Results

Cleaned AG News test set: 6,357 headlines, none of which occurs in training. Both main methods were tuned on a separate development set and the test set was used once.

| Test set | TF-IDF + LR | MiniLM + LR | MiniLM + 10-NN *(extra)* |
|---|---:|---:|---:|
| Accuracy | 86.49 | 82.95 | **88.38** |
| Macro-F1 | 86.25 | 82.69 | **88.16** |
| F1 Business | 82.58 | 77.36 | **83.96** |
| F1 Sci/Tech | 83.89 | 80.28 | **86.14** |
| Business → Sci/Tech confusions | 139 | 191 | 126 |
| Sci/Tech → Business confusions | 125 | 155 | 130 |
| **Business ↔ Sci/Tech total** | 264 | 346 | **256** |

* **Answer for the approved methods: no.** Frozen MiniLM embeddings with logistic regression make 31% more Business ↔ Sci/Tech confusions than TF-IDF (346 vs 264) and score lower on every class. The gap is significant (McNemar p = 1.6 × 10⁻⁷ on Business/Sci-Tech headlines; bootstrap intervals exclude 0).
* **Unfamiliar words:** on headlines that contain a word outside the TF-IDF vocabulary, the two methods are statistically tied (p = 0.93 on Business/Sci-Tech headlines).
* **Extra experiment** (added after the error analysis, not in the proposal): with a nearest-neighbour vote instead of logistic regression, the *same frozen embeddings* beat TF-IDF (88.2 vs 86.3 macro-F1, p = 1.3 × 10⁻⁶). The weakness is the linear classifier, not the pretrained representation.

See Section 4 of the notebook for the statistical tests, the challenge-factor analysis and five explained errors.

## How to run

### Google Colab (recommended)
1. Open `headline_topic_classifier.ipynb` in Colab (*File → Open notebook → GitHub*, paste this repository's URL).
2. Optional but faster: *Runtime → Change runtime type → T4 GPU*. MiniLM encoding takes about 1 minute on a GPU and 10 to 20 minutes on a CPU.
3. *Runtime → Run all.* The first cell installs anything that is missing. The data are downloaded from Kaggle automatically; no Kaggle account is needed.

To use the **saved models** in Colab without retraining, put this repository's URL in `REPO_URL` in the first code cell. The notebook then clones the repository, and **5. Try it** works after running only **0. Setup**.

### Local Jupyter
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter lab headline_topic_classifier.ipynb
```
Then *Run → Run All Cells*. With the embeddings cached, a full run takes about 6 minutes on a laptop CPU; the first run takes about 15 minutes. `requirements.txt` pins the exact versions used for the saved results (Python 3.13).

### Try it (no retraining)
Section **5. Try it** loads `models/tfidf_logreg.joblib` and `models/minilm_logreg.joblib`. Type a headline into the `headline` field and run the cell. It prints the topic from the selected model (TF-IDF + LR, which had the higher development macro-F1), the probability of each topic, and the other model's answer. Empty input gives a helpful message instead of an error.

## Data and preprocessing
* **Source:** [AG News (Kaggle, amananandrai)](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset), 120,000 training and 7,600 test articles, balanced over 4 classes. Only the **Title** column and the label are used. Fallback download: the original CSV on GitHub ([mhjabreel/CharCnn_Keras](https://github.com/mhjabreel/CharCnn_Keras)).
* **Cleaning:**
  * repair HTML leftovers (`#39;` → `'`, `quot;`, `&lt;b&gt;...&lt;/b&gt;`, `\$`) in 15,318 training titles;
  * remove publisher tags such as `(Reuters)`, `(AP)`, `(PC World)` and `(Update1)`, which leak the label, from 16,849 titles;
  * drop 17 `#NAME?` spreadsheet-error rows.
* **Fair split:** duplicates are detected on the cleaned, normalised text. 1,149 training titles with conflicting labels (2,724 rows) are removed as label noise. Also removed: 1,235 test titles that also occur in training, and 3 test titles (6 rows) with conflicting labels. The 90/10 train/dev split is made over unique headlines, stratified by label, so copies of a headline stay in one split. Final sizes: **train 105,569 · dev 11,692 · test 6,357**, with no headline shared between splits.

## Methods
| | Method 1 (traditional) | Method 2 (pretrained) |
|---|---|---|
| Features | TF-IDF word uni+bigrams (100,466 features, `min_df=2`, sublinear tf) | frozen `all-MiniLM-L6-v2` sentence embeddings (384-d), standardised |
| Classifier | multinomial logistic regression (`saga`) | multinomial logistic regression (`lbfgs`) |
| Tuned on dev | n-gram range, C → (1,2), C = 8 (dev macro-F1 87.15) | C → 0.003 (dev macro-F1 83.35) |

## Repository layout
```
headline_topic_classifier.ipynb   main notebook (Setup, Data, Method 1, Method 2, Results, Try it, Sources)
requirements.txt                  package versions used for the saved results
models/                           saved models used by "Try it" (no retraining needed)
results/metrics.json              every number reported in the report and slides
results/*.csv                     dev tuning tables, test predictions, our own example headlines
results/figures/*.png             figures used in the report and slides
data/processed/                   cleaned train/dev/test splits written by the notebook
deliverables/                     report, slides and appendix for the Blackboard submission
data/raw/, cache/                 downloaded CSVs and cached embeddings (re-created, not committed)
```

## Reproducibility notes
* Fixed seed (42); the split, TF-IDF model, tuning and bootstrap are deterministic.
* The saved results were produced on a CPU. Embeddings computed on a GPU can differ in the last decimal places, which may flip a handful of borderline MiniLM predictions.
* Thread counts are capped in Setup. On some multi-core laptops the default thread settings made logistic regression dozens of times slower.

## Credits
* Data: AG News, X. Zhang, J. Zhao and Y. LeCun (NeurIPS 2015), built from A. Gulli's AG corpus of news articles.
* Model: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache-2.0); Reimers and Gurevych (EMNLP 2019); Wang et al. (NeurIPS 2020).
* Libraries: scikit-learn, pandas, NumPy, SciPy, PyTorch, sentence-transformers, kagglehub, matplotlib, joblib.
* **AI assistance:** parts of the code and text were drafted with the help of an AI assistant (Claude, Anthropic). The team reviewed, ran and checked everything and is responsible for the content.

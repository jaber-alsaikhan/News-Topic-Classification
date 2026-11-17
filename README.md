# News Topic Classification from Headlines

The notebook takes one English news headline and predicts World, Sports, Business or Sci/Tech. We compare TF-IDF + logistic regression with all-MiniLM-L6-v2 sentence embeddings + logistic regression.

**Question:** If only the headline is available, is a pretrained sentence-embedding classifier better than TF-IDF logistic regression at separating Business and Sci/Tech headlines?

## Results (test set, 6,469 headlines)

| | TF-IDF + LR | MiniLM + LR |
|---|---|---|
| Accuracy | 86.86% | 82.95% |
| Macro F1 | 86.67% | 82.72% |
| F1 Business | 82.75% | 76.77% |
| F1 Sci/Tech | 85.17% | 81.01% |
| Business <-> Sci/Tech confusions | 258 | 360 |

Answer: no, TF-IDF separates Business and Sci/Tech better when only the headline is used.

## How to run

We ran the notebook on Google Colab with a GPU runtime. It also runs in Jupyter, but the embedding step is much slower without a GPU.

1. Open `headline_topic_classifier.ipynb` in Jupyter or Google Colab.
2. Run all cells from top to bottom (Jupyter: Kernel > Restart & Run All, Colab: Runtime > Run all).
3. The first cell installs `sentence-transformers` and `kagglehub`, and the data is downloaded from Kaggle automatically.
4. To test a new headline, use the Try it section (run the Setup and Data cells first).

Package versions we used (Google Colab): numpy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, torch 2.14.1, sentence-transformers 5.7.0.

## Files

- `headline_topic_classifier.ipynb`: the notebook (Setup, Data, Method 1, Method 2, Results, Try it, Sources)
- `models/`: the saved models
- `results/`: test results, predictions and the confusion matrix figure
- `deliverables/`: report, slides and appendix

## Sources

- AG News dataset, Kaggle (amananandrai/ag-news-classification-dataset), Zhang et al. 2015
- sentence-transformers/all-MiniLM-L6-v2 (Hugging Face)
- pandas, numpy, scikit-learn, sentence-transformers, PyTorch, matplotlib


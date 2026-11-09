# News Topic Classification from Headlines

NLP Project - Option 3 | Section 1 | Group 2 | Imam Abdulrahman Bin Faisal University

| Role | Name | Student ID |
|---|---|---|
| Team leader | Mohammed Saeed Alamodi | 2240009007 |
| Member 2 | Abdulaziz Mohammed Alkhlawi | 2240006095 |
| Member 3 | Jaber Saleh Alsaikhan | 2240006539 |

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

## Files

- `headline_topic_classifier.ipynb`: the notebook (Setup, Data, Method 1, Method 2, Results, Try it, Sources)
- `models/`: the saved models
- `results/`: test results, predictions and the confusion matrix figure
- `deliverables/`: report, slides and appendix

## Sources

- AG News dataset, Kaggle (amananandrai/ag-news-classification-dataset), Zhang et al. 2015
- sentence-transformers/all-MiniLM-L6-v2 (Hugging Face)
- pandas, numpy, scikit-learn, sentence-transformers, PyTorch, matplotlib
- We used an AI assistant (Claude) to help write parts of the code.

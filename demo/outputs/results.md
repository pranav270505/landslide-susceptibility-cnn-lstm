| Model | Accuracy | Precision | Recall | F1 | AUC | Spec. spatial neg | Spec. temporal neg |
|---|---|---|---|---|---|---|---|
| CNN-LSTM (hybrid) | 0.913 | 0.86 | 0.925 | 0.892 | 0.956 | 0.818 | 1.0 |
| CNN-only | 0.689 | 0.562 | 0.9 | 0.692 | 0.778 | 0.848 | 0.233 |
| LSTM-only | 0.738 | 0.614 | 0.875 | 0.722 | 0.848 | 0.333 | 1.0 |
| Random Forest | 0.913 | 0.844 | 0.95 | 0.894 | 0.979 | 0.788 | 1.0 |
| MLP | 0.874 | 0.787 | 0.925 | 0.851 | 0.954 | 0.697 | 1.0 |
| SVM | 0.903 | 0.857 | 0.9 | 0.878 | 0.975 | 0.818 | 1.0 |
| Decision Tree | 0.874 | 0.8 | 0.9 | 0.847 | 0.929 | 0.727 | 1.0 |

Test set: 103 samples (40 landslide, 33 spatial negatives, 30 temporal negatives), held-out spatial blocks. 'Spec.' columns = share of that kind of negative correctly rejected.

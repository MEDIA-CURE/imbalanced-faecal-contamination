# Machine learning methods for imbalanced data set for prediction of faecal contamination in beach waters

**[Overview](#overview)**
| **[Abstract](#abstract)**
| **[Citation](#citation)**

## Overview

The code in this repository will allow you to reproduce the experiments from the paper as well as use the code for your own projets 

## Abstract

Predicting water contamination by statistical models is a useful tool to manage health risk in recreational beaches. Extreme contamination events, i.e. those exceeding normative are generally rare with respect to bathing conditions and thus the data is said to be imbalanced. Modeling and predicting those rare events present unique challenges. Here we introduce and evaluate several machine learning techniques and metrics to model imbalanced data and evaluate model performance. We do so by using a) simulated data-sets with known characteristics and b) a real data base with records of faecal coliform abundance monitored for 10 years in 21 recreational beaches in Uruguay (N ≈ 19000) using *in situ* and meteorological variables. We discuss advantages and disadvantages of the methods and provide a simple guide to perform models for a general audience. We also provide R codes to reproduce model fitting and testing. Most machine learning techniques are sensitive to imbalance and require specific data pre-treatment (e.g. upsampling) to improve performance. The use of metrics other than accuracy (number of hits over total cases) is suggested as accuracy does not capture the model performance on the rare class. True positive rates (TPR) and False positive rates (FPR) are recommended instead. Among the 52 possible candidate algorithms tested, the stratified Random Forest presented the better performance improving TPR in 50% with respect to baseline (0.4) and outperforming baseline in the evaluated metrics. Support vector machines combined with upsampling method or synthetic minority oversampling technique (SMOTE) performed well, similar to Adaboost with SMOTE. This results suggests that combining modeling strategies is necessary to improve our capacity to anticipate water contamination. Stratified Random Forest is an efficient technique to deal with imbalanced data set that showed an exceptional performance.

## Citation

If you find this code helpful for your work please cite our Paper as: 

```bibtex
@article{,
  title={Machine learning methods for imbalanced data set for prediction of faecal contamination in beach waters},
  author={},
  journal={},
  year={2021}
}
```
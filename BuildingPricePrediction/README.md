## Objective
To conduct descriptive analysis, statistical testing and predict residential building sales price

## Approach
### Statistical testing
Chi-square test (for two categorical variables), t-test (for one binary and one numerical variables), AVOVA (for one multi-categorical and one numerical variables)

### Price prediction
Kernel ridge regression, SVM, Decision tree, Ensemble models

## Results
Training dataset: random selection of 80%

Test dataset: remaining 20%

### RMSE on test dataset:
- Linear kernel regression: 158
- Polynomial kernel regression: 197
- RBF kernel regression: 427
- SVM: 770
- Decision tree: 296
- Random forest: 244
- Gradient boosting: 169


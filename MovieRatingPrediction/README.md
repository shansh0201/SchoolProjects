## Objective
To predict movie ratings given by the users

## Approach
Model: Naive models (taking mean), Matrix factorization, Collaborative filtering (KNN)

## Results
Training dataset: MovieLens (random selection of 90%)

Test dataset: MovieLens (remaining 10%)

### RMSE on test data
- Global mean: 1.116
- Movie-based mean: 0.980
- User-based mean: 1.199
- Matrix factorization: 0.977
- User-based collaborative filtering: 0.913
- Movie-based collaborative filtering: 0.852

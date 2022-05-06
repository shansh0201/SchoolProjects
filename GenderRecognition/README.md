## Objective
To use facial images to predict gender

## Approach
Model: Logistic regression, Ensemble of logistic regressions

Optimizers: Gradient descent, SGD, Momentum, ADAM

Feature engineering: Pixel intensity rescaling, Image resizing, Black-and-white conversion, Image cropping

## Results
Training dataset: CelebA (first 5000)

Test dataset: CelebA (last 5000)

### Changing size of training dataset
![image](https://user-images.githubusercontent.com/100197692/167084692-775f53ca-5f33-44e2-9acb-20251dbb865c.png)

### Changing image resolutions
![image](https://user-images.githubusercontent.com/100197692/167084948-298ad9c2-8a44-4108-82f8-ff9b9bcd77f7.png)

### Model ensembling
![image](https://user-images.githubusercontent.com/100197692/167085132-d45b13f2-843f-446d-b0a4-103365033a3c.png)





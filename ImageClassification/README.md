## Objective
To predict the class of images from 15 selected scenes

## Approach
Model: CNN

Feature engineering: Adding noise, Data augmentation

Improvements: adding a supplementary CNN model for images from the predicted classes with low precision

## Results
Training dataset: random selection of 64% from 1500 images consisting of 15 classes (namely, bedroom, coast, forest, highway, industrial, insidecity, kitchen, livingroom, mountain, office, opencountry, store, street, suburb, tallbuilding)

Validation dataset: remaining 16% of the total (for hyper-parameters tuning and early stopping)

Test dataset: remaining 20% of the total

![image](https://user-images.githubusercontent.com/100197692/167099105-1b60ef68-ae07-4552-8116-a3407fd45995.png)

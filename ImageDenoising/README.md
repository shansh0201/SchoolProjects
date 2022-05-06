## Objective
To reproduce "Deep Convolutional Dictionary Learning" algorithm for image denoising and improve the model architecture for denoising the images with other noise distributions.

## Approach
"Deep Convolutional Dictionary Learning" algorithm

Improvements: Adding a 3x3 median filter followed by a 3x3 Gaussian blurring filter in pre-processing stage.

## Results
Train dataset: WED

Test dataset: URBAN100

![image](https://user-images.githubusercontent.com/100197692/167082257-2f6e2e1b-a54c-41a2-9b53-f3de25f957a2.png)


## Reference
Hongyi Zheng, Hongwei Yong, and Lei Zhang. Deep convolutional dictionary learning for image denoising. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), pages 630–641, June 2021.

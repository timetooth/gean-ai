# Unsupervised Ancient Document Image Denoising Based on Attention Mechanism

# Motivation

The digitization of ancient documents is on the rise, while the poor quality of raw manuscripts creates problems for researchers and readers. Thus, we hope to propose a method to reduce noise on images and improve their quality. However, paired ancient document images are almost non-existent. Therefore, we proposed a model that could be trained on unpaired images.

<img src="https://github.com/timetooth/gean-ai/blob/main/pic/origin_ancient.png" width=50% height=50%>

# Dataset
  We chose Fangshan Shijing as our data source and cropped 1200 positive and negative 256 * 256 patches each. The ratio of training set to test set is 5:1. Here are two samples:
  
 <img src="https://github.com/timetooth/gean-ai/blob/main/pic/samples.jpg" width=70% height=70%>

 Download : [Noise2Denoise](https://drive.google.com/drive/folders/19_nighq97KBlwxTtan3t0Kf6qqCUsZIj?usp=sharing)
  
|Name| Explaination|
|------|------|
|trainA | clean images for training|
|trainB | noisy images for training|
|testA | clean images for testing|
|testB | noisy images for testing|
|testBgt_sim.txt | character level annotation for testB|
|testBgt_tra.txt | character level annotation for testB|

The pdf file is the data source from which we have extracted all the samples. You can use it as you like. 
# Network Architecture
<img src="https://github.com/timetooth/gean-ai/blob/main/pic/generator.jpeg" width=70% height=70%>

Our improvements focus on the generator module, which works by embedding the attention module in the stacked residuals module. We hope to focus the feature map on the foreground or background of the image for the purpose of denoising but not changing the text. We have tried two different attention mechanisms: SE and CBAM. The former focuses on the channel dimension only, while the latter focuses on both the channel dimension and the spatial dimension.

  
# Results
<img src="https://github.com/timetooth/gean-ai/blob/main/pic/denoise_result.png" width=100% height=100%>

## Attention Map Visualization
<img src="https://github.com/timetooth/gean-ai/blob/main/pic/attention_map.png" width=70% height=70%>

## FID metrics
| Feature Dimention  | 64 | 192    | 2048  |
|--------|------------|-------   |--------|
| CycleGAN | 1.34    | 8.95   | 66.47 | 
| CycleGAN+CBAM | 3.16    | 11.52   | 67.18 |
| CycleGAN+SE | **1.06**    | **4.85**   | **59.79** |
<img src="https://github.com/timetooth/gean-ai/blob/main/pic/loss.jpeg" width=100% height=100%>

# Configurations
torch 1.9.1

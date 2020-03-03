# Overview

<img src="/images/1.png" width = "100"> <img src="/images/2.png" width = "100"> <img src="/images/3.png" width = "100">
<img src="/images/4.png" width = "100"> <img src="/images/5.png" width = "100"> <img src="/images/6.png" width = "100">
<img src="/images/7.png" width = "100"> <img src="/images/8.png" width = "100">


[Convolution Neural Network for German Traffic Sign Recognition](https://github.com/prateeshreddy/GTSRB_road_sign_recog) 


# Dataset
We use [GTSRB dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads)
 
 * 43 classes
 * More than 50,000 images in total
 * Large, lifelike database
 * Reliable ground-truth data due to semi-automatic annotation
 * Physical traffic sign instances are unique within the dataset

# Data Preprocessing
The usual preprocessing in this case would include scaling of pixel values to [0, 1] (as currently they are in [0, 255] range), representing labels in a one-hot encoding and shuffling. Looking at the images, histogram equalization may be helpful. 

[Histogram Equalization](https://en.wikipedia.org/wiki/Histogram_equalization) is a computer image processing technique used to improve contrast in images . It accomplishes this by effectively spreading out the most frequent intensity values, i.e. stretching out the intensity range of the image

# Data Augmentation
The amount of data we have is not sufficient for a model to generalise well. It is also fairly unbalanced, and some classes are represented to significantly lower extent than the others. But we will fix this with data augmentation!

### Flipping
First, we are going to apply a couple of tricks to extend our data by flipping. You might have noticed that some traffic signs are invariant to horizontal and/or vertical flipping, which basically means that we can flip an image and it should still be classified as belonging to the same class. Some signs can be flipped either way — like Priority Road or No Entry signs, other signs are 180° rotation invariant, and to rotate them 180° we will simply first flip them horizontally, and then vertically. Finally there are signs that can be flipped, and should then be classified as a sign of some other class. This is still useful, as we can use data of these classes to extend their counterparts. We are going to use this during augmentation, and this simple trick lets us extend original 39,209 training examples to 63,538, nice! And it cost us nothing in terms of data collection or computational resources.

### Rotation and projection
However, it is still not enough, and we need to augment even further. After experimenting with adding random rotation, projection, blur, noize and gamma adjusting, I have used rotation and projection transformations in the pipeline. Projection transform seems to also take care of random shearing and scaling as we randomly position image corners in a [±delta, ±delta] range.   

# Data Pipeline
Building the input pipeline in a machine learning project is always long and painful, and can take more time than building the actual model.Using tf.data to build efficient pipelines for images.

The Dataset API allows you to build an asynchronous, highly optimized [Data Pipeline](https://cs230.stanford.edu/blog/datapipeline/#building-an-image-data-pipeline) to prevent your GPU from data starvation. It loads images from the disk, applies optimized transformations, creates batches and sends it to the GPU. Former data pipelines made the GPU wait for the CPU to load the data, leading to performance issues.

# Result
After 20 epochs this model scored 98.7% accuracy on the test set, which is not too bad. As there was a total of 12,630 images that we used for testing, apparently there are 85 examples that the model could not classify correctly.

Signs on most of those images either have artifacts like shadows or obstructing objects. There are, however, a couple of signs that were simply underrepresented in the training set.

In conclusion, according to different sources human performance on a similar task varies from 98.3% to 98.8%, therefore this model seems to outperform an average human. Which, I believe, is the ultimate goal of machine learning!

           
# References

1. [Traffic Sign Recognition with Multi-Scale CNN Paper By Yann LeCun and Pierre Sermanet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
1. [Building a data pipeline Using Tensorflow tf.data By Andrew Ng and Kian Katanforoosh](https://cs230.stanford.edu/blog/datapipeline/)
1. [Recognising Traffic Signs Using Deep Learning By Eddie Forson](https://towardsdatascience.com/recognizing-traffic-signs-with-over-98-accuracy-using-deep-learning-86737aedc2ab) an article from towardsdatascience
           




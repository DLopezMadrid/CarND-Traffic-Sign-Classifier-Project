#**Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./readme_imgs/histogram_1.png "histogram"
[image2]: ./readme_imgs/training_imgs_disp.png "Random training imgs"
[image3]: ./readme_imgs/new_distrib_augmented_imgs.png "new histogram"
[image4]: ./readme_imgs/augmented_imgs.png "Augmented images"
[image5]: ./readme_imgs/new_imgs.png "New Images"
[image6]: ./readme_imgs/softmax_top5.png "Top5"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You can find my code for this project here: [project code](https://github.com/DLopezMadrid/CarND-Traffic-Sign-Classifier-Project)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy to solve this part of the project (Cell #2), these are the results obtained:

* Number of training examples = 39209
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43
* Class count = [ 210 2220 2250 1410 1980 1860  420 1440 1410 1470 2010 1320 2100 2160  780
  630  420 1110 1200  210  360  330  390  510  270 1500  600  240  540  270
  450  780  240  689  420 1200  390  210 2070  300  360  240  240]

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

Code for this section can be found on Cells #3 to #7
Next, you can see a histogram for the different categories and their image count
![alt text][image1]
It can be seen how the original dataset does not have a uniform distribution of the classes. That is not necessarly a problem and probably is due to the natural ocurrance of the different types of traffic signs

Here you can see a random sample of different images from this dataset
![image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code and associated functions for this step can be found on cells #8 to #14.

The first step is to generate additional images in order to augment the original training dataset. This will make the network less reliant on the position of the traffic signs on the image and the different perspectives. Using OpenCV functions, the code generates images with a random value of rotation, translation and shearing inside some defined ranges.

Based on some tests that I run, I chose to generate a uniform set of data (same number of samples for every class). This is done so the system does not have bias towards certain signals that are more common on the training data.

![image3]

Then, the augmented dataset is converted into grayscale. I have found that using grayscale images yields better results for this problem and as a side benefit it also helps the system to train faster. This aspect makes sense to me since there is not much information coded into the color of the traffic signs, usually is used as a tool to improve contrast and as a redundant source of information. On the other hand, working with color images can also be counterproductive if the images have different white balance levels or contrasts.

After this step, the images are fed to a function that includes a random variation in brightness in them to reduce the dependency of the network on this type of information.

Finally, the data is normalized in order to avoid the use of large values that can destabilize the system

The next image shows a random sample of the resultant images after being through these processes:

![image4]

After this, I put the test data through the same pre processing in order to have coherent data to test the network.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The training data is shuffled and then splitted into training and validation in Cell #15. The test data comes directly from the pickled files. The sklearn module contains useful functions to do these tasks

As mentioned in the previous section, the code for data augmentation is stored together with the code for data preprocessing in Cells #8 to #14



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model is based on the LeNet network previously used in the course with some modifications. The model code can be found in Cell #17

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout					| keep_prob = 0.7
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x16  	|
| RELU					|												|
| Dropout					| keep_prob = 0.7
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 1x1     	| 1x1 stride, VALID padding, outputs 5x5x64 	|
| RELU					|												|
| Dropout					| keep_prob = 0.5
| Fully Connected	      	| flatten previous output (input = 1600),  output = 400 				|
| Dropout					| keep_prob = 0.7 |
| Fully Connected	      	| input = 400,  outputs = 120 				|
| Dropout					| keep_prob = 0.7 |
| Fully Connected	      	| input = 120,  outputs = 43 				|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I define the required functions and train my model between Cell #18 and Cell #22
Based on trial and error I found the hyperparameters that I am using. For the learning rate I chose 0.001, for the regularization weight 0.00001, for the dropout keep probability 0.7 and for the batch size 128 (the biggest that I could fit on my GPU memory).

Regarding the optimizer, I reused the one chosen for the previous project (Adam)


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in Cell #23 of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.978
* test set accuracy of 0.924

I started by using the same basic LeNet model as in previous lessons of the course. This model had acceptable performance but due to its simplicity it under fitted the data, so a more complex model was needed. I tried to improve the LeNet model by adding a 1x1 convolution layer that I thought that may be able to add more flexibility to the network. The addition of this layer improved the results greatly.

Then, I started experimenting with the use of grayscale vs RGB images, seeing a noticeable improvement by using grayscale ones. After this, I introduced some methods to avoid overfitting the data, including regularization and dropout, achieving even better results.

All the hyperparameters were tuned as described before in order to achieve over 90% accuracy. The number of epochs was kept at 10 due to the training times and the computing power of my GPU (Nvidia 960M GTX)

It is important to remark that due to the random nature of the augmented data and initialisation of the model, the accuracy values can vary (from my results, from 90 to 92 % approx.)


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Instead of using just 5 images for this section I chose to use a higher number (24) to have a better granularity and be able to experiment with more variants.

Here you can see the chosen images with the ground truth (top) and predicted label (bottom)

![alt text][image5]

This dataset contains 16 images that are part of the classes that the network has been trained for (label starts with "german_")
It also include 8 additional images on classes that the system has not seen before (label starts with "alt_")

This way we can test the accuracy of the model in expected situations (trained classes) and unexpected/new situations (new classes). Some of the new classes are very similar to the ones present in the training dataset but with slight variations. A good example of this is the "alt_road_work" sign, in this case, the image is flipped along the Y axis. Another case is the "alt_ped_xing" sign, in the training dataset there are pedestriang signs but they do not count with the zebra crossing on them. "alt_iguana_crossing" and "alt_elephant_crossing" have a very similar shape and color to priority road too.

Inside the trained classes ones, it is worth mentioning the blur and angle of "german_stop_3" and the different contrast levels of "german_60_kmh" and "german_60_kmh_2"
####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located between Cell #28 and Cell #30

The results of the prediction can be seen on the image of the previous section. The ground truth for each image is on top of each one and the predicted label is below each one.

The model was able to correctly guess 12 of the 16 traffic signs that had classes on the training dataset, getting an accuracy of 75%, lower than with the testing dataset but caused in part by the low number of samples. From the other 8 images that are not part of the original classes, it was only capable of classifying correctly one (yield) of the three that are variations of the original classes (yield sign from the US, pedestrian with zebra crossing and roadworks flipped in Y axis)

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the last three cells of the notebook

Here you can see an image with the different results:

![image6]

It is important to remark that the images that do not belong to the original classes show high probabilities for multiple predicted classes.

The system gets quite confused with the speed limits signals, it may not be flexible enough to do the text recognision that is necessary to classify those correctly. Straight ahead or left suffers from a similar condition.

Also, the two double curve signs, although very similar for the human eye (some slight variation in the geometry), they look quite different for the model.

The "german_stop_3" sign is too distorted for the system to recognise it properly (stop is not even in the top5)

The "alt_ped_xing" makes some interesting and almost understandable predictions (road narrows right and road work)

As expected, the elephant crossing gets highly classified as priority road, although not as top option (top option is roundabout mandatory)

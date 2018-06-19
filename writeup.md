# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[signs]: ./imgs/signs.png "signs"
[dist1]: ./imgs/dist1.png "train dist"
[dist2]: ./imgs/dist2.png "valid dist"
[dist3]: ./imgs/dist3.png "test dist"
[sign_bright]: ./imgs/sign_bright.png "sign_bright"
[sign_contrast]: ./imgs/sign_contrast.png "sign_contrast"
[sign_org]: ./imgs/sign_org.png "sign org"
[sign_sharp]: ./imgs/sign_sharp.png "sign sharp"
[sign_dataaug]: ./imgs/sign_dataaug.png "sign dataaug"
[test_imgs]: ./imgs/test_imgs.png "test_imgs"
[good_dist]: ./imgs/good_dist.png "good dist"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

![alt text][signs]

Here is an exploratory visualization of the data set. These histograms show the distributions of labels in each dataset(train, validation, test).

Three distributions looks almost same shape. But, there is a bias in the distributions.
This means this dataset have the number of label 2(Speed limit (50km/h)) is so large. However, the number of label 37(Go straight or left) looks so small amount of data. In train data, the number of label 2 is 2010 and the number of label 37 is 180. It is a big bias.
So learning label 37 looks difficult. 

![alt text][dist1]
![alt text][dist2]
![alt text][dist3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to do only one thing as preprocess. It is normalization because image data should have mean zero and equal variance.
I used quick way, (pixel - 128) /128 for normalization.
I do not convert the images to grayscale because it did not increase validation score.

I decided to generate additional data because there is a bias in the data distributions.
To do remove a bias, I especially generated the labels of small amount data. 

To add more data to the the data set, I used the following techniques.
* brightness_transform
  * This transformation changes the image brightness randomly.  
  ![alt text][sign_org]
  ![alt text][sign_bright]  
* contrast_transform
  * This transformation changes the image contrast randomly.  
  ![alt text][sign_org]
  ![alt text][sign_contrast]  
* sharp_transform
  * This transformation changes the image sharpness randomly.  
  ![alt text][sign_org]
  ![alt text][sign_sharp]  
* translation
  * This transformation shifts the image location randomly.
* rotate_transform
  * This transformation rotates the image randomly.
* scale_transform
  * This transformation change the image scale randomly.

Here is an example of an original image and an augmented image:

  ![alt text][sign_org]
  ![alt text][sign_dataaug]  

And I get a better distribution of train dataset as a following image.  

  ![alt text][good_dist]  

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 						|
| RELU | |
| flatten| outputs 400|
| dropout ||
| Fully connected		|  outputs 120       									|
| Fully connected		|  outputs 84       									|
| Fully connected		|  outputs 43       									|
| Softmax				|         									||
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam as optimizer because it is known as good choice on the first try. But recent year, [this research](https://arxiv.org/abs/1705.08292) says Adaptive optimization may be not better than SGD. So, we should consider other optimizer in practice. However, I have no time to do that.

Next, I used batch_size=128 because I have enough memory to run this batch_size. And I used number of epoch=100 because I considered it is enough epoch for this simple model.

Finally, I used learning rate=0.001. I tried to change it, but the validation score did not improve.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.961 
* test set accuracy of 0.950

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * I choosed [Lenet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) because I learned it in this lecture and it is a simple DNN model. So I can change it easily.
* What were some problems with the initial architecture?
  * Lenet-5 do not have any dropout layer. So the network will overfit easily. I need to add dropout layers.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * I used validation accuracy to adjust architecture. But architecture decrease low validation accuracy.
* Which parameters were tuned? How were they adjusted and why?
  * I changed the position of dropout layers because it strongly affects validation accuracy. And I changed the number of augumented data. I think It is also affects validation accuracy.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * I think dropout layers are important to improve validation accuracy. Lenet-5 without dropout layer overfit train data easily. It is not good property for better model. If I add dropout layer, It worked well. 

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][test_imgs]

The first image might be easy to classify because it is clear and not rotated image.  
The second image might be difficult to classify because it looks dark color and has little complex human shape.  
The third image might be difficult to classify because it looks small and rotated image.  
The fourth image might be easy to classify because it looks clear color.  
The fifth image might be difficult to classify because it looks rotated image and has black spaces on the side.  


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout mandatory      		| Roundabout mandatory   									| 
| Pedistrians     			| Speed limit(20km/h)										|
| Turn right ahead			| Turn right ahead											|
| Speed limit(70km/h)	      		| Speed limit(70km/h)					 				|
| Stop			| Stop      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. It is far from test set accuracy. I think these new images looks higher quality than test dataset. So the accuracy may be different. And Pedistrians sign has little complex shape. So it might be difficult.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is really sure that this is a Roundabout mandatory (probability of 1.00). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Roundabout mandatory   									| 
| 0.00     				| Road narrows on the right 										|
| 0.00					| Traffic signals											|
| 0.00	      			| Childeren crossing					 				|
| 0.00				    |  Speed limit (80km/h)     							|


For the second image, the model is sure that this is Speed limit sign. It is a wrong prediction. Speed limit sign similar with pedistrians signs.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.67         			| Speed limit(20km/h)   									| 
| 0.17     				| Speed limit(30km/h) 										|
| 0.16					| Speed limit(70km/h)											|
| 0.00	      			| General caution					 				|
| 0.00				    | Speed limit(50km/h)      							|

For the third image, the model is really sure that this is a Turn right ahead (probability of 1.00). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Turn right ahead   									| 
| 0.00     				|  Turn left ahead										|
| 0.00					| Roundabout mandatory											|
| 0.00	      			| Vehicles over 3.5 metric tons prohibited					 				|
| 0.00				    | Ahead only    							|

For the fourth image, the model is really sure that this is a Speed limit sign (probability of 1.00). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			|  Speed limit(70km/h)  									| 
| 0.00     				|  Speed limit(20km/h)										|
| 0.00					| Speed limit(30km/h)											|
| 0.00	      			| Go straight or left					 				|
| 0.00				    | Rondabout mandatory    							|

For the fifth image, the model is really sure that this is a Stop sign (probability of 1.00). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			|  Stop  									| 
| 0.00     				|  No entry										|
| 0.00					| Yield											|
| 0.00	      			| No vehicles					 				|
| 0.00				    | No passing    							|



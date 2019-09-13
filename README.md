# Vehicle Detection and Tracking

The steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector.
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

More details of the steps are in VehicleDetection.ipynb

---
### Parameters Tuning for HOG features and color space

I used linear SVM classifier for classifying car and non-car images. Firstly, I set the parameters for HOG (orientations, pixels per cell, cells per block) to give reasonable accuracy on linear SVM and then try all the different color space (RGB, HSV, LUV, HLS, YUV, YCrCb) to compare the accuracy. I chose YCrCb color space since it gives the highest accuracy among all the color spaces. I increased and decreased the HOG parameters slightly to see if it improve the accuracy of the classifier and it turns out that the HOG parameters I initially used, gives the highest accuracy. (detailed experimentation in VehicleDetection.ipynb)

### Sliding Window

After training the classifier, I have to use sliding window to search for cars in an image. First, I set a region of interest to exclude the sky as it will speed up the classifier and reduce false positive as cars will not be found on that region. Instead of changing the sliding window size, I used a sliding window of 64x64 (the size of training data trained with linear SVM) and scale the image to 1.5 to downsample the image to essentially have the effect of using a 96x96 (64*1.5) sliding window. I have set the windows overlap to 0.75 as it seems to improve the performance. As the classifier is already performing slowly, I decided not to use another sliding window of different size. HOG is applied to the region of interest and then run through the sliding window instead of applying HOG on each of the sliding window. This improved the performance significantly.

### Implementation on Video

Since vehicles will appear around the same area on consecutive frames, the classifier should take into account of that for classifying the vehicles. I have created a class for heatmap to update the values for every frames in the video. There are 2 parameters for the method; threshold and heatmap decay rate. Threshold will help to remove or reduce false positives by ignoring areas with lower heat. Heatmap decay rate is to adjust the amount of information to keep over the frames. If the heatmap decay rate is 0, the classifier does not take into account of previous frames while if the heatmap decay rate is 1, the classifier will take into account of every frames before the current frame.

---

### Pipeline (video)

Here's a [link to my video result](https://youtu.be/OZOAll-wQow)

---

### Discussion

To improve the reliability of the classifier, I might want to do a hard negative mining by applying the classifier on many images and taking examples that are classified wrongly. The training images are all taken when the weather condition is good and therefore it will be better if I augment the dataset by added noises such as rain, changing the brightness, etc. 1 of the problems faced is tuning the parameters. In the future, I will probably apply my classifier on images with true labels (bounding box on vehicle known) and try different parameters to improve performance. If the video is taken in bad weather condition or taken at night, it will probably fail. 

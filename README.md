# KNN

KNN is a supervised learning classifier, which uses proximity/Eucledian distance to make classifications or predictions about the
grouping of an individual data point.
K in KNN is the number of nearest neighbors (based on the chosen distance metric).
The quality of the predictions depends on the distance measure.
Therefore, the KNN algorithm is suitable for applications for which sufficient domain knowledge is available.
The prediction in the KNN is made based on the MODE. 

For example, there are 30 emplyees in a company. Suppose there are
30 records such that there are 20 1’s ( staying in a company) and 10 0’s (leaving a company). So, we can conclude for a new
associate that the prediction is 1.
It depends on how many neighbors you want to assign for a new data. Based on it, the data will be classified as 0 or 1.
It plots the data into the axes to classify a group based on the k-nearest neighbor idea.






# Different steps of K-NN for classifying a new data point:

Step 1: Select the value of K neighbors(say k=4)

Step 2: Find the K (4) nearest data point for our new data point based on euclidean distance, Manhattan distance.

Step 3: Among these K data points count the data points in each category

Step 4: Assign the new data point to the category that has the most neighbors of the new datapoint.

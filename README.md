# Machine Learning Algorithms:

An algorithm is a procedure used for solving a problem or performing a computation.

An algorithm is a set of commands that must be followed for a computer to perform calculations or other problem-solving operations.

There are four types of machine learning algorithms: supervised,  unsupervised, Semi-supervised and reinforcement.


![image](https://github.com/Tiwari666/KNN/assets/153152895/43ddb527-fa12-49cc-855c-3e8b03139dfb)





![image](https://github.com/Tiwari666/KNN/assets/153152895/fcd3f086-b7ea-4443-89c6-63087a9007e5)


The main distinction between the supervised and unsupervised algorithms is the use of labeled datasets. To put it simply, supervised learning uses labeled input and output data, while an unsupervised learning algorithm does not.

# Supervised (labels are present for all the observations) vs. unsupervised learning ( no labels are present for all the observation in the dataset)

In supervised learning, the algorithm “learns” from the training dataset by iteratively making predictions on the data and adjusting for the correct answer. While supervised learning models tend to be more accurate than unsupervised learning models, they require upfront human intervention to label the data appropriately. For example, a supervised learning model can predict how long your commute will be based on the time of day, weather conditions and so on. But first, one needs to train the model to know whether the rainy weather extends the driving time.

Unsupervised learning models, in contrast, work on their own to discover the inherent structure of unlabeled data. Note that they still require some human intervention for validating output variables. For example, an unsupervised learning model can identify that online shoppers often purchase groups of products at the same time. However, a data scientist must validate it to makes sense for a recommendation engine for grouping
 baby clothes with an order of diapers, applesauce and sippy cups.


 


# Unsupervised learning

Unsupervised learning uses machine learning algorithms to analyze and cluster unlabeled data sets. These algorithms discover hidden patterns in data without the need for human intervention (hence, they are “unsupervised”).

Unsupervised learning models are used for three main tasks: clustering, association and dimensionality reduction:

A) Clustering is a data mining technique for grouping unlabeled data based on their similarities or differences. For example, K-means clustering algorithms assign similar data points into groups, where the K value represents the size of the grouping and granularity. This technique is helpful for market segmentation, image compression, etc.

B) Association is another type of unsupervised learning method that uses different rules to find relationships between variables in a given dataset. These methods are frequently used for market basket analysis and recommendation engines, along the lines of “Customers Who Bought This Item Also Bought” recommendations.

C) Dimensionality reduction is a learning technique used when the number of features  (or dimensions) in a given dataset is too high. It reduces the number of data inputs to a manageable size while also preserving the data integrity. Often, this technique is used in the preprocessing data stage, such as when autoencoders remove noise from visual data to improve picture quality.


# Semi-supervised learning (: The best of both worlds)

If we cannot decide on whether to use supervised or unsupervised learning algorithms, a Semi-supervised learning is a happy way to go, where we use a training dataset with both labeled and unlabeled data. It’s particularly useful when it’s difficult to extract relevant features from data — and when we have a high volume of data.

In many practical situations, the cost to label is quite high, since it requires skilled human experts to do that. So, in the absence of labels in the majority of the observations but present in few, semi-supervised algorithms are the best candidates for the model building. These methods exploit the idea that even though the group memberships of the unlabeled data are unknown, this data carries important information about the group parameters.

Semi-supervised learning is ideal for medical images, where a small amount of training data can lead to a significant improvement in accuracy. For example, a radiologist can label a small subset of CT scans for tumors or diseases so the machine can more accurately predict which patients might require more medical attention.


# Deep Learning vs Reinforcement Learning:


Deep learning learns from real-world data, reinforcement learning learns from synthetic data as an agent interacts with an environment, receiving feedback based on its actions.

Synthetic data/Algorithmically generated data/ not real data:

Synthetic data is an information that's artificially generated rather than produced by real-world events. 
Typically created using algorithms, synthetic data can be deployed to validate mathmatical models and to train machine learning models.

# Area of Reinforcement learning

Reinforcement learning models make predictions by getting rewards or penalties based on actions of agents performed within an environment. 

 A reinforcement learning system generates a policy that defines the best strategy for getting the most rewards.

Reinforcement learning is used to train robots to perform tasks, like walking around a room, and software programs like AlphaGo to play the game of Go.

 
Robotics: Robots learn to perform tasks in the physical world.

Video gameplay: Teaches bots to play video games.

Resource management: Helps enterprises plan allocation of resources.

# KNN
KNN (k-Nearest Neighbors) is a supervised machine learning algorithm. In supervised learning, the algorithm learns from labeled training data, where each data point is associated with a known label or outcome. KNN specifically requires labeled training data to classify new data points based on their similarity to existing data points with known class labels. Therefore, it falls under the category of supervised learning algorithms.

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



###############################################################################
# Sources: Various Online documents
Link1: https://towardsdatascience.com/types-of-machine-learning-algorithms-you-should-know-953a08248861

Link2: https://www.ibm.com/blog/supervised-vs-unsupervised-learning/

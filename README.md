# Credit_Risk_Analysis
# Overview

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Credit risk is associated with the possiblity of a client failing to meet contractual obligations. Therefore, we will need to employ different techniques to train and evaluate models with unbalanced classes. We are using imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we will oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, we will use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, we will compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once we are done, we will evaluate the performance of these models and build a recommendation for the best machine learning model to use for credit risk predictions.

# Results
## Naive Random Oversampling

In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.

![](Resources/Naive_0.png?raw=true)

![](Resources/Naive_1.png?raw=true)

![](Resources/Naive_2.png?raw=true)

Balanced Accuracy Score: 65.40%  
Precision : The precision is low for High risk loans and is very good for Low risk loans.  
Recall : 0.66 for High risk and .65 for Low risk. 0.65 sensitivity.  
F1 score: avg/total is 0.79, So it is good at predicting Low risk loan classes.

## SMOTE Oversampling

The Synthetic Minority Oversampling Technique (SMOTE) is another oversampling approach to deal with unbalanced dataset. In SMOTE, like random oversampling, the size of the minority is inreased. In SMOTE, new instances are interpolated. For an instance from the minority class, a number of its closest neighboors is chosen and the new values are created.

![](Resources/SMOTE_1.png?raw=true)

![](Resources/SMOTE_2.png?raw=true)

Balanced Accuracy Score: 63.69%  
Precision : The precision is low for High risk loans and is very good for Low risk loans.  
Recall : 63% for High risk and 64% for Low risk. Overall 64% sensitivity.  
F1 score: avg/total is 0.78, So it is good at predicting Low risk loan classes.

## Cluster Centroids (Undersampling)
Undersampling is another technique to address class imbalance. Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased.

Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class. Here in example both high-risk and low-risk categories counts to 260.



![](Resources/Cluster_centroid_1.png?raw=true)

![](Resources/Cluster_centroid_2.png?raw=true)

Balanced Accuracy Score: 63.69%  
Precision : The precision is low for High risk loans and is very good for Low risk loans.  
Recall : 63% for High risk, 64% for Low risk and 64% sensitivity.
F1 score: avg/total is 0.78 So it is good at predicting Low risk loan classes.

## Combination (Over and Under) Sampling

SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN is a two-step process:

* Oversample the minority class with SMOTE.
* Clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.

![](Resources/SMOTEENN_1.png?raw=true)

![](Resources/SMOTEENN_2.png?raw=true)

Balanced Accuracy Score: 63.69%  
Precision : The precision is low for High risk loans and is very good for Low risk loans.  
Recall : 63% for High risk, 64% for Low risk and 64% sensitivity.  
F1 score : avg/total is 0.78 So it is good at predicting Low risk loan classes.


## **Ensemble Learners**
## Balanced Random Forest Classifier

The imblearn.ensemble module include methods generating under-sampled subsets combined inside an ensemble. BalancedRandomForestClassifier is another ensemble method in which each tree of the forest will be provided a balanced bootstrap sample
A balanced random forest randomly under-samples each boostrap sample to balance it.


![](Resources/Ensemble_0.png?raw=true)

![](Resources/BalancedRandomForestClassifier_1.png?raw=true)


Balanced Accuracy Score: 78.44%  
Precision : The precision is low for High risk loans and is very good for Low risk loans.  
Recall : 68% for High risk, 89% for Low risk and 89% sensitivity.  
F1 score: avg/total is 0.94, So it is good at predicting Low risk loan classes.

## Easy Ensemble Classifier
A specific method which uses AdaBoostClassifier as learners in the bagging classifier is called “EasyEnsemble”. The EasyEnsembleClassifier allows to bag AdaBoost learners which are trained on balanced bootstrap samples.


![](Resources/EasyEnsembleClassifier_1.png?raw=true)


Balanced Accuracy Score: 92.43%  
Precision : The precision is low for High risk loans and is very good for Low risk loans.  
Recall : 91% for High risk, 94% for Low risk and 94% sensitivity.  
F1 score: avg/total is 0.96, but it is good at predicting Low risk loan classes.

## Summary

One way of validating model's performance: its accuracy score. The accuracy score is a quick indicator of how accurate each of the models was at predicting the loan status.

Naive Random Oversampling : Accuracy Score is 65.40%  
SMOTE Oversampling : Accuracy Score is 63.69%  
Cluster Centroids (Undersampling) : Accuracy Score is 63.69% 
Combination (Over and Under) Sampling : Accuracy Score is 63.69%  
Balanced Random Forest Classifier: Accuracy Score is 78.44% 
Easy Ensemble Classifier: Accuracy Score is 92.43%

Precision, also known as positive predictive value (PPV). Precision is obtained by dividing the number of true positives (TP) by the number of all positives (i.e., the sum of true positives and false positives, or TP + FP).
Compared to all other models, Easy Ensemble classifier is better in identifying high risk loans category that is of 7%.


Based on the analysis results, I recommend Easy Ensemble classifier Model over all other models. It appears that the *Easy Ensemble Classifier model* has the best balance out of all other models on the given dataset because of it's high accuracy score(92.43%), a good balance of precision and recall scores.





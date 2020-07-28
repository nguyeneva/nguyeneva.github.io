---
layout: post
title: Comparing Machine Learning Models
subtitle:
tags: [R programming, RStudio, KNN, LDA, QDA, statistical modeling, machine learning, supervised learning, confusion matrix, F1 score, logloss]
---

### Which of the 3 models is the best for this data?
Based on the evaluation metrics below, I would choose the k-nearest neighbors (K=3) method as the 'best' for this data. The reason is KNN outperforms for all metrics when comparing to LDA and QDA. The KNN has the highest values for Sensitivity, Recall, Precision, Specificity, and F1 Score. The KNN has the lowest values for Misclassficiation Rate and LogLoss.  

### Data Set
The data set, `bank`, is from the gclus package in R. The data set contains six measurements made on 100 genuine Swiss banknotes and 100 counterfeit ones.

**Columns:**
- **Status:** 0 = genuine, 1 = counterfeit
- **Length:** Length of bill, mm
- **Left:** Width of left edge, mm
- **Right:** Width of right edge, mm
- **Bottom:** Bottom margin width, mm
- **Top:** Top margin width, mm
- **Diagonal:** Length of image diagonal, mm

### LDA Evaluation Metrics
{% highlight r linenos %}
# Load Relevant Libraries
library(gclus)
library(MASS)
library(MLmetrics)

# Load Data
data(bank)

# Model with LDA
bank.lda<-lda(Status~.-Bottom,data=bank, CV=TRUE)

# Evaluation Metrics
Sensitivity<-Sensitivity(bank$Status, bank.lda$class)
Recall<-Recall(bank$Status, bank.lda$class) #same thing!
Precision<-Precision(bank$Status, bank.lda$class)
Specificity<-Specificity(bank$Status, bank.lda$class)
F1_Score<-F1_Score(bank$Status, bank.lda$class)
LogLoss<-LogLoss(bank.lda$posterior[,2], bank$Status)
Misclassification_Rate<-1-mean(bank$Status==bank.lda$class)

# Print Metrics
print(paste("Sensitivity:", Sensitivity), quote=FALSE)
print(paste("Recall:", Recall), quote=FALSE)
print(paste("Precision:", Precision), quote=FALSE)
print(paste("Specificity:", Specificity), quote=FALSE)
print(paste("F1_Score", F1_Score), quote=FALSE)
print(paste("LogLoss:", LogLoss), quote=FALSE)
print(paste("Misclassification_Rate:", Misclassification_Rate), quote=FALSE)
{% endhighlight %}

**Output:**
```
[1] Sensitivity: 0.98
[1] Recall: 0.98
[1] Precision: 1
[1] Specificity: 1
[1] F1_Score 0.98989898989899
[1] LogLoss: 0.067725242554878
[1] Misclassification_Rate: 0.01
```
### QDA Evaluation Metrics
{% highlight r linenos %}
# Model with QDA
bank.qda <- qda(Status~.-Bottom, data=bank, CV=TRUE)

# Evaluation Metrics
Sensitivity<-Sensitivity(bank$Status, bank.qda$class)
Recall<-Recall(bank$Status, bank.qda$class) #same thing!
Precision<-Precision(bank$Status, bank.qda$class)
Specificity<-Specificity(bank$Status, bank.qda$class)
F1_Score<-F1_Score(bank$Status, bank.qda$class)
LogLoss<-LogLoss(bank.qda$posterior[,2], bank$Status)
Misclassification_Rate<-1-mean(bank$Status==bank.qda$class)

# Print Metrics
print(paste("Sensitivity:", Sensitivity), quote=FALSE)
print(paste("Recall:", Recall), quote=FALSE)
print(paste("Precision:", Precision), quote=FALSE)
print(paste("Specificity:", Specificity), quote=FALSE)
print(paste("F1_Score", F1_Score), quote=FALSE)
print(paste("LogLoss:", LogLoss), quote=FALSE)
print(paste("Misclassification_Rate:", Misclassification_Rate), quote=FALSE)
{% endhighlight %}

**Output:**
```
[1] Sensitivity: 0.98
[1] Recall: 0.98
[1] Precision: 1
[1] Specificity: 1
[1] F1_Score 0.98989898989899
[1] LogLoss: 0.093952319798637
[1] Misclassification_Rate: 0.01
```
### Model with KNN
{% highlight r linenos %}
# Load Relevant Libraries
library(class)

# Model with KNN
bank.knn<-knn.cv(subset(bank, select=-c(Bottom)), bank$Status, k=3, prob=TRUE)

# Evaluation Metrics
Sensitivity<-Sensitivity(bank$Status, bank.knn)
Recall<-Recall(bank$Status, bank.knn) #same thing!
Precision<-Precision(bank$Status, bank.knn)
Specificity<-Specificity(bank$Status, bank.knn)
F1_Score<-F1_Score(bank$Status, bank.knn)


probs <- attr(bank.knn, "prob")
missedprobs <- 1-probs[bank$Status!=bank.knn]
missedprobs[missedprobs==0] <- 1e-3
probs[probs==0] <- 1e-3
LogLoss<-(sum(-log(probs[bank$Status==bank.knn])) + sum(log(missedprobs)))/200

Misclassification_Rate<-1-mean(bank$Status==bank.knn)

# Print Metrics
print(paste("Sensitivity:", Sensitivity), quote=FALSE)
print(paste("Recall:", Recall), quote=FALSE)
print(paste("Precision:", Precision), quote=FALSE)
print(paste("Specificity:", Specificity), quote=FALSE)
print(paste("F1_Score", F1_Score), quote=FALSE)
print(paste("LogLoss:", LogLoss), quote=FALSE)
print(paste("Misclassification_Rate:", Misclassification_Rate), quote=FALSE)
{% endhighlight %}

**Output:**
```
[1] Sensitivity: 0.99
[1] Recall: 0.99
[1] Precision: 1
[1] Specificity: 1
[1] F1_Score 0.994974874371859
[1] LogLoss: 0.0345387763949107
[1] Misclassification_Rate: 0.005
```

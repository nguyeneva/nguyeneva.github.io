---
layout: post
title: Comparing Machine Learning Models
subtitle:
tags: [R programming, RStudio, KNN, LDA, QDA, Confusion Matrix, F1 Score, LogLoss]
---

### Data Set
`bank` from the gclus package in R. Six measurements made on 100 genuine Swiss banknotes and 100 counterfeit ones.

Columns:  
- Status: 0 = genuine, 1 = counterfeit
- Length: Length of bill, mm
- Left: Width of left edge, mm
- Right: Width of right edge, mm
- Bottom: Bottom margin width, mm
- Top: Top margin width, mm
- Diagonal: Length of image diagonal, mm


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
{% highlight r linenos %}
{% endhighlight %}

{% highlight r linenos %}
{% endhighlight %}

![png](/assets/img/CNN_files/CNN_files_3.png)   

The full jupyter notebook can be found on my GitHub [here](https://github.com/nguyeneva/projects/blob/master/image_classification_cnn.ipynb).

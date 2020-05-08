---
layout: post
title: Data Manipulation with Natural Language Processing
subtitle:
tags: [data manipulation, python, NLTK, pandas, numpy, matplotlib]
---


import pandas as pd
data=pd.read_csv("/users/evanguyen/Previous Courses/data_542/data542_finalproject/cleaned_data.csv")

```python
len(data)
```


966691

### 1. Loading Relevant Libraries and Data

First, import relevant libraries.

{% highlight python linenos %}
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
{% endhighlight %}


I have the files stored on my local machine. I use `glob` to only pull review filenames.

{% highlight python linenos %}
files1=glob.glob('/users/evanguyen/data_542/Dataset/**/*reveiws*.csv', recursive=True)
files2=glob.glob('/users/evanguyen/data_542/Dataset/**/*reviews*.csv', recursive=True)

files_total=files1+files2
{% endhighlight %}


I create an empty list, read each csv into a data frame, and append each data frame into the list.

{% highlight python linenos %}
list1 = []

for file_ in files_total:
    df1 = pd.read_csv(file_,index_col=None, header=0)
    list1.append(df1)
{% endhighlight %}

I want to label each app by their categories name. I will need to get the categories name from the filename using `split` and `__contains__`.

{% highlight python linenos %}
categories=[]
for i in range(0,len(files_total)):
    if (files_total[i][-40:].split("_",3)[-1].__contains__('FREE')) & ((files_total[i][-40:].split("_",3)[-1].__contains__('TOP_FREE')==False)) :
        categories.append(files_total[i][-40:].split("_",4)[-1][:-9])
    elif files_total[i][-40:].split("_",3)[-1].__contains__('TOP_FREE'):
        categories.append(files_total[i][-40:].split("_",5)[-1][:-9])
    else:
        categories.append(files_total[i][-40:].split("_",3)[-1][:-9])
{% endhighlight %}

After retrieving the categories name I add it to the data frames within the list.

{% highlight python linenos %}
for i in range(0,len(files_total)):
    list1[i]['category']= categories[i]
{% endhighlight %}

Merge all the reviews files into one data frame using `pandas concat`.

{% highlight python linenos %}
for file_ in files_total:
    review_df = pd.concat(list1)
{% endhighlight %}

### 2. Merging Two Data Sets

```python
##################### pulling only detailed data set #####################
files3=glob.glob('/users/evanguyen/data_542/Dataset/**/*all_detailed*.csv', recursive=True)

list2 = []

#####################  reading each csv filename into a dataframe #####################
for file_2 in files3:
    dfcolumns = pd.read_csv(file_2,
                        nrows = 1)

    df2 = pd.read_csv(file_2,
                  header = None,
                  skiprows = 1,
                  usecols = list(range(len(dfcolumns.columns))),
                  names = dfcolumns.columns)


    list2.append(df2)

##################### getting categories from filename #####################
categories2=[]
for i in range(0,len(files3)):
    if (files3[i][-40:].split("_",3)[-1][:-9].__contains__('FREE'))==False :
        categories2.append(files3[i][-40:].split("_",3)[-1][:-9])
    elif (files3[i][-40:].split("_",3)[-1][:-9].__contains__('FREE')==True) & (files3[i][-40:].split("_",3)[-1][:-9].__contains__('TOP_FREE')==False)  :
        categories2.append(files3[i][-40:].split("_",4)[-1][:-9])
    elif (files3[i][-40:].split("_",3)[-1][:-9].__contains__('TOP_FREE')==True)  :
        categories2.append(files3[i][-40:].split("_",5)[-1][:-9])
    else:
        categories2.append(files3[i][-40:].split("_",3)[-1][:-9])

# adding date from filename to data frame
for i in range(0,len(files3)):
    list2[i]['date_import']= files3[i][34:44]        

##################### adding categories name from filename to data frame #####################
for i in range(0,len(files3)):
    list2[i]['category']= categories2[i]

##################### merging all the reviews files #####################
for file_2 in files3:
    detailed_df = pd.concat(list2)

##################### creating unique IDs #####################
detailed_df['uniqueID']=range(1,len(detailed_df)+1)
detailed_df=detailed_df.set_index(['uniqueID'])
```


### 3. Data Cleaning
Drop all duplicate rows.

{% highlight python linenos %}
review_df=review_df.drop_duplicates()
without_duplicates=len(review_df)
{% endhighlight %}

Add unique IDs to the data frame and set it as the index.

{% highlight python linenos %}
review_df['uniqueID']=range(1,len(review_df)+1)
review_df=review_df.set_index('uniqueID')
{% endhighlight %}

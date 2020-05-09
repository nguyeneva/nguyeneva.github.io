---
layout: post
title: Data Manipulation with Natural Language Processing
subtitle:
tags: [data manipulation, python, NLTK, pandas, numpy, matplotlib]
---

The project's purpose is to clean and manipulate text data prior to applying machine learning techniques.

### Dataset Description
Two csv files are collected for mobile applications each week. One csv file includes the reviews and metadata for the reviews, and the other csv file includes details of each app.  

The datasets are collected for approximately 80 mobile applications. The datasets live on a Google Drive [here](https://drive.google.com/drive/folders/1j1YdI5IVaK0PUHmZTMsTpOWXcnm8m2d7?usp=sharing).

Below is a sample screenshot of the datasets' file structure after downloading it from Google Drive.

### Step 1. Load Relevant Libraries and Datasets

First, import relevant libraries.

{% highlight python linenos %}
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
{% endhighlight %}

The files are stored within multiple folders on my local machine, and `glob` makes it easy to only pull review filenames. Some filenames had 'reviews' misspelled as 'reveiws', which required two glob commands.

![png](/assets/img/data_manipulation/folder_structure.png)

{% highlight python linenos %}
files1=glob.glob('/Dataset/**/*reveiws*.csv', recursive=True)
files2=glob.glob('/Dataset/**/*reviews*.csv', recursive=True)

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

Some of the filenames have different naming conventions, which required the first two complicated clauses in the if statement.

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

Drop all duplicate rows.

{% highlight python linenos %}
review_df=review_df.drop_duplicates()
without_duplicates=len(review_df)
{% endhighlight %}


Lastly, I add a unique ID index to the data set.

{% highlight python linenos %}
review_df['uniqueID']=range(1,len(review_df)+1)
review_df=review_df.set_index('uniqueID')
{% endhighlight %}

Results of the final data frame.
{% highlight python linenos %}
review_df.head()
{% endhighlight %}

![png](/assets/img/data_manipulation/head_1.png)


### 2. Merging Two Data Sets

Similar steps as above to pull only csv files with review metadata using `glob`. I read each csv into a data frame then into a list, add the categories name from the filename, merge all the data frames into one data frame, and add unique ID as the index.

```python
##################### pulling only detailed data set #####################
files3=glob.glob('/Dataset/**/*all_detailed*.csv', recursive=True)

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

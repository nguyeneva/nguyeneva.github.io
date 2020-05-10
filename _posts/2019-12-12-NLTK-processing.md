---
layout: post
title: Data Manipulation with Natural Language Processing
subtitle:
tags: [data manipulation, python, NLTK, pandas, numpy, matplotlib]
---

The project's purpose is to clean and manipulate text data prior to applying machine learning techniques. The dataset contains errors to simulate data collection in the real world.   

### Dataset Description
Two csv files are collected for mobile applications each week. One csv file includes application reviews, and the other file includes application details.  I'm only interested in the reviews csv files.    

The datasets are collected for approximately 80 mobile applications. The datasets live on a Google Drive [here](https://drive.google.com/drive/folders/1j1YdI5IVaK0PUHmZTMsTpOWXcnm8m2d7?usp=sharing).   

Below is a sample screenshot of the datasets' file structure after downloading it from Google Drive. The dates represents when the data was pulled from the Google Play Store.

![png](/assets/img/data_manipulation/folder_structure.png)  


### Step 1. Load Relevant Packages and Data

First, import relevant libraries.   

{% highlight python linenos %}
import glob
import pandas as pd
import numpy as np
import nltk
from itertools import groupby
import matplotlib.pyplot as plt
{% endhighlight %}

The csv files live in multiple folders broken out by dates of data pull. `glob` makes it easy to only pull review filenames from each folder. Some filenames had 'reviews' misspelled as 'reveiws', which required two glob commands.   

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

I convert all the string fields to lowercase for aggregation later.   

{% highlight python linenos %}
review_df=review_df.apply(lambda x: x.astype(str).str.lower())
{% endhighlight %}

Drop all duplicate rows.   

{% highlight python linenos %}
review_df=review_df.drop_duplicates()
without_duplicates=len(review_df)
{% endhighlight %}

I add a unique ID index to the data set.   

{% highlight python linenos %}
review_df['uniqueID']=range(1,len(review_df)+1)
review_df=review_df.set_index('uniqueID')
{% endhighlight %}

Results of the final data frame.   

{% highlight python linenos %}
review_df.head()
{% endhighlight %}

![png](/assets/img/data_manipulation/head_1.png)

### 2. Data Manipulation
I create a new column with my text manipulation called "text_mod".    

I remove non-ASCII characters using `encode("ascii", "ignore")`.

{% highlight python linenos %}
review_df['text_mod']=review_df['text'].map(lambda x: str(x).encode("ascii", "ignore").decode('utf-8'))
{% endhighlight %}

I remove punctuations using regular expressions.   

{% highlight python linenos %}
review_df['text_mod'] = review_df['text_mod'].str.replace(r'[^\w\s]+', '')
{% endhighlight %}

There are application reviews with emojis and some are in other languages. I create a function to calculate the ratio of non-English words for each review. If the ratio is higher than 50%, I drop the review observation (not the entire row).   

I use `nltk.corpous.words.words()` to create an English words dictionary for the comparison. I use `nltk.wordpunct_tokenize()` to separate each word by a space into a list.

Please note, if you have not downloaded nltk package before, it'll require an additional command, `nltk.download()` after importing the package.   

{% highlight python linenos %}
words = set(nltk.corpus.words.words())
def english(string):
    wordslist=nltk.wordpunct_tokenize(string)
    true_count=0
    false_count=0
    for w in wordslist:
        if w in words or not w.isalpha():
            true_count+=1
        if w not in words:
            false_count+=1
    if false_count!=0 or true_count!=0:
        if (false_count/(false_count+true_count)>=0.50):
            return "NOTENGLISHDROP"
        else:
            return string

review_df.text_mod=review_df.text_mod.apply(lambda x: english(x))
review_df.drop(review_df[review_df.text_mod == 'NOTENGLISHDROP'].index, inplace=True)
{% endhighlight %}


I remove words that have been misspelled with more than 2 characters.   

{% highlight python linenos %}
def extra_char(string):
    if string is not None and len(string)>0:
        count=0
        wordslist=nltk.wordpunct_tokenize(string)
        for w in wordslist:
            if w.lower() in words or not w.isalpha():
                return string
            if w.lower() not in words:
                groups = groupby(w)
                count = [sum(1 for _ in group) for label, group in groups]
                for c in count:
                    if c>2:
                        string=string.replace(str(w),'')
                        string=string.strip()  
                        return string
    else:
        return string

review_df.text_mod=review_df.text_mod.apply(lambda x: extra_char(x))
{% endhighlight %}

I remove reviews that contain two or less number of words.   

{% highlight python linenos %}
word_count = review_df['text_mod'].str.split().str.len()
review_df=review_df[~(word_count<=2)]
{% endhighlight %}

# Kaggle_Disaster_Tweets


## Introduction

This notebook is the final delivery for the lecture "DataScience SS2020". 
This notebook was also used to participate in the following Kaggle Challenge on NLP ([LINK](https://www.kaggle.com/c/nlp-getting-started/overview)). 
The goal was the interpretation and analysis of the data, preprocessing and the creation of an NLP-ML-model and subsequent prediction. The model was then applied to a submission data set and submitted several times (after applying different models and hyperparameter optimizations).
The notebook was created directly in Kaggle and is therefore optimized to run in Kaggle.

**My Kaggle ID:** Benedikt Merkel

**My best Kaggle Score:** 0,84155


## Summary




**This notebook represents two notebooks in one:**

**The first part uses the following pre-trained BERT model:**

* bert_en_uncased_L-24_H-1024_A-16/2 [LINK](https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2)
* The process is accelerated by the GPU.



**The second part uses the following pre-trained BERT model:**

* DistilBERT
* bert-base-cased
* bert-large-cased
* bert-large-cased-whole-word-masking-finetuned-squad

(You can find the pretrained models here: [LINK](https://huggingface.co/transformers/pretrained_models.html))
The process was accelerated with the TCU.


## Summary

  
  
**Part 1: bert_en_uncased_L-24_H-1024_A-16/2**

* Imports and check for accelerator unit
* Import the Data 
* Data Observation
* Data Cleansing
* Building the Model
* Train the Model
* Results Part 1
    
 **Part 2: bert-large-cased, and others**   
 
* Imports and check for accelerator unit
* Import the Data 
* Data Observation
* Data Cleansing
* Building the Model
* Hyperparameters
* Train the Model
* Results Part 2


 **Overall Conclusion**




## My general procedure

At first i tried to work with destilBERT as well as bert-base-cased, bert-large-cased, bert-large-cased-whole-word-masking-finetuned-squad and the TCU accelerator, which succeeded. Although the Accuracy, the Validation Accuracy and the Validation loss seemed better than in part 1, i could just achieve  akaggle score of max. 0.8271, even though i achieved better validation accuracies (0,88) and better validation losses (ca. 0,01). While i worked with the different pretrained models, i noticed that my score was approx. 0.01 points better with the large models than with the smaller counterpart.
Because of the experiance, i decided to go directly for the large model in part 1 instead of first trying the smaller one, like i did in the 2nd part (which was chronological the first). 

More details about the scores and parameters can be found under the respective parts.


   
   
   
## RESULTS PART 1


1. With Cleasning, 8 Epochs, batchzise: 16, validation split: 0.2, DropOut:  0      = kaggle score: 0.84155 
2. Without Cleansing, 8 Epochs, batchzise: 16, validation split: 0.2, DropOut:  0      = kaggle score: 0.80135
3. With Cleasning, 4 Epochs, batchzise: 16, validation split: 0.2, DropOut:  0      = kaggle score: 0.83849 
4. With Cleasning, 4 Epochs, batchzise: 16, validation split: 0.3, DropOut:  0      = kaggle score: 0.83083
5. With Cleasning, 16 Epochs, batchzise: 16, validation split: 0.2, DropOut:  0      = kaggle score: 0.83358
6. With Cleasning, 8 Epochs, batchzise: 16, validation split: 0.2, DropOut:  0.1      = kaggle score: 0.83450
   
 
## RESULTS PART 2

Strangly the accuracies, val. accuracies and validation losses are a lot better with this models, but somehow the kaggle score stays unter 0.8271.

Some examples from the best run with every pretrained model:

1. destilBERT: 0.75268 (data cleaned; can't remember the hyperparameters)
2. bert-base-cased: 0.85000 (data cleaned; can't remember the hyperparameters)
3. bert-large-cased-whole-word-masking-finetuned-squad: 0.82715 (strangly without data cleaning; Batchsize: 64, Steps: 100, Epochs: 4,  max_len: 160)
4. bert-large-uncased: 0.82562 (with data cleaning; Batchsize: 16, Steps: 150, Epochs: 8, max_len: 160) 
   
   
   
## OVERALL CONCLUSION

First of all: the challenge was very interesting. Since I had not worked with NLPs in the labs before, it was a special challenge. Therefore it was helpful to partly use existing solutions like the pre-trained BERT models and to be able to view notebooks of other participants of the challenge. However, I did not get very clever with my results from Part 2. As already mentioned I applied this model first and tested nearly 50 different combinations of models and hyperparameters. Even though the values of the model (validation accuracy, validation loss, and accuracy) were extremely good, the best score I could get was only 0.8271. The only advantage of these models was that they were relatively fast to calculate even with a large number of steps. 

The bert_en_uncased_L-24_H-1024_A-16/2 model, on the other hand, took much longer (sometimes 20 times longer), but also gave a better Kaggle Score. However, the values (validation accuracy, validation loss, and accuracy ) of the calculated models were a bit worse.


## Inpirations:


* https://www.kaggle.com/raenish/cheatsheet-text-helper-functions
* https://www.kaggle.com/datafan07/disaster-tweets-nlp-eda-bert-with-transformers
* https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras/data
* https://www.kaggle.com/sagar7390/nlp-on-disaster-tweets-eda-glove-bert-using-tfhub
* http://education.abcom.com/detecting-slang-using-bert/
* https://analyticsindiamag.com/tutorial-on-keras-callbacks-modelcheckpoint-and-earlystopping-in-deep-learning/
* https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

# POS-Tagging-of-Tweets

Twitter (www.twitter.com) is a popular microblogging site that contains a large amount of user-generated posts, each called a tweet. By analyzing the sentiment embedded in tweets, we gain valuable insights into commercial interests such as the popularity of a brand or the general belief in the viability of a stock1. Many companies earn revenue from providing such tweet sentiment analysis. In such sentiment analysis, a typical upstream task is the Part-of-Speech (POS) tagging of tweets, which generates POS tags that are essential for other downstream natural language processing tasks.

## This project aims to build a POS tagging system using the Hidden Markov Model (HMM). 

twitter_train.txt is a labelled training set. Each non-empty line of the labeled data contains one token and its associated POS tag separated by a tab. An empty line separates sentences (tweets). Below is an example with two tweets (there is an empty line after “?? ,”).
Both twitter_dev_no_tag.txt and twitter_dev_ans.txt have the same format as twitter_train.txt, except that the former does not contain tags and the latter does not contain tokens.
twitter_dev_ans.txt contains the tags of the corresponding tokens in twitter_dev_no_tag.txt. 
twitter_train*.txt and twitter_dev*.txt contain 1101 and 99 tweets respectively.
twitter_tags.txt contains the full set of 25 possible tags.

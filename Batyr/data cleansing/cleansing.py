# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 19:43:15 2018

@author: bnuryyev
"""
import re
import csv
import pandas as pd


## Functions for analysis

def readCSVIntoList(link):
    """
        Reads CSV to list using 'csv.reader()'
                
        Params:
            link (String) - contains link to the file
        
        Returns:
            tweets_list (List) - containing the tweets in vector form (m x 1)
    """
    
    tweets_file = csv.reader(open(link, "rt", encoding="cp1252"))
    tweets_list = list(tweets_file)    

    return tweets_list



def cleanUpTweets(tweets_list):
    """
        Cleans the tweets list using regular expressions.
        Removes URIs and ampersands. Does not allow tweets
        of size less than 15.
        
        Params:
            tweets (List) - contains list (vector) of tweets
        
        Returns:
            clear_tweets (List) - clean list of tweets
    """
    # Constants
    MIN_TWEET_SIZE  = 30
    REGEX_URI       = '(http|https):\S+[/a-zA-Z0-9]'
    REGEX_AMPERSAND = '(&amp;)'
    REGEX_NEWLINE   = '\n'
    REGEX_EMOJIS    = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    
    # Filtered list
    clear_tweets = []
    
    # (ugly) Loop through the list 
    for lst in tweets_list:
        inner_list = []
        for t in lst:
            if (len(t) >= MIN_TWEET_SIZE):
                t = re.sub(REGEX_URI, '', t, flags=re.MULTILINE)
                t = re.sub(REGEX_AMPERSAND, ' and ', t, flags=re.MULTILINE)
                t = REGEX_EMOJIS.sub('', t)
                t = re.sub(REGEX_NEWLINE, '', t, flags=re.MULTILINE)
            
            # Check tweet size after "brushing up"
            if (len(t) >= MIN_TWEET_SIZE):
                inner_list.append(t)
        if len(inner_list) != 0:
            clear_tweets.append(inner_list)
    
    
    # Return
    return clear_tweets
    



def main():
    # Open and read CSV file
    tweets_list = readCSVIntoList('trump_tweets_raw.csv')
    
    # Clean up the tweets
    tweets_list_cleared = cleanUpTweets(tweets_list)    
    
    # Save to a file
    pd.DataFrame(tweets_list_cleared).to_csv('trump_tweets_cleaned.csv', index=False, header=False)
    
    


if __name__ == "__main__":
    # execute only if run as a script
    main()

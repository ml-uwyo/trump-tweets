# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:36:57 2018

@author: bnuryyev
"""

import numpy as np
import pandas as pd


# Markov chain class
class MarkovChain(object):
    def __init__(self, text, n_grams, min_len):
        self.grams = []
        self.grams_possibilities = {}
        self.grams_count = {}
        self.n_grams = n_grams
        self.text = text.split(" ")
        self.min_len = min_len
        self.nGramsWords()
        self.nGramsCount()
        self.nGramsPossibilities()
        
        
    def nGramsWords(self):
        """ Get the n-grams as list of lists """
        for i in range(len(self.text) - self.n_grams + 1):
            self.grams.append(self.text[i : i + self.n_grams])    


    def nGramsCount(self):
        """ Compute the frequency of each gram """
        for i in range(len(self.text) - self.n_grams + 1):
            # Get the gram first
            gram = self.grams[i]
            
            # Count the frequency using dictionary of lists as keys
            if not repr(gram) in self.grams_count:
                self.grams_count[repr(gram)] = 1
            else:
                self.grams_count[repr(gram)] += 1
                            
    
    def nGramsPossibilities(self):
        """ Get the transition probabilities for each state. 
            Appends the possible words into array inside dictionary.
            The more of the same word X - the more the probability of
            the word X to get generated
        """
        for i in range(len(self.text) - self.n_grams):
            # Get the gram
            gram = self.grams[i]
            
            # Get the next possible word
            next_word = self.text[i + self.n_grams]
            
            # Initialize array if the gram is not there yet
            if not repr(gram) in self.grams_possibilities:
                self.grams_possibilities[repr(gram)] = []
            
            # Otherwise append the word to already existing gram in the dict
            self.grams_possibilities[repr(gram)].append(next_word)            
    
    
    # TODO: plots the frequency of each gram
    #def plotFrequency():
        #return
    
                
    def generateText(self):
        """ Generates the text randomly choosing 
            from the probability dictionary 
        """                
        # Here you have two choices: start from the beginning of a text
        # or somewhere randomly in the text.
        # Currently starts somewhere randomly.
        current_i = np.random.choice(len(self.grams))
        current = self.grams[current_i]
        
        # Append the current gram
        output = " ".join(current)
        
        # Initialize variable to store the next word
        next_word = ""
        
        for i in range(self.min_len):            
            possibilities = self.grams_possibilities[repr(current)]
            
            # If there are no possibilities, skip to the next iteration
            if len(possibilities) == 0:
                continue
            else:
                next_word = np.random.choice(possibilities)
            
            # Append the word to the so far generated text
            output = output + " " + str(next_word)
            
            # Get the next gram
            current_i = current_i + 1
            current = self.grams[current_i]
        
        
        return output


# Load data (uncomment line below to use Trump tweets data)
#data = pd.read_csv('trump_data.csv', header=None, na_values=['.'], encoding='cp1252')
        
# Open Robinson Crusoe file to read the text
f = open("robinsonCrusoe.txt", "r")

# Read the file (comment the line below to use Trump tweets data)
data = f.read()

# Original data in np.array format
data_array = np.array(data)

# Make the letters lowercase and concatenate into a single string
data_array = data_array.flatten()
data_array = [x.lower() for x in data_array]
data_array = ' '.join(data_array)

# Instantiate Markov chain class
n_grams = 3
text_size = 100
markov = MarkovChain(data_array, n_grams, text_size)
output = markov.generateText()
print(output)
# trump-tweets

Repo for Batyr

## Data

`trump_tweets.csv` contains data made by Talon

`trump_tweets_raw.csv` contains data made by me with a help of [Trump Twitter Archive](http://www.trumptwitterarchive.com/archive)

`trump_tweets_cleaned.csv` contains data cleaned up using regex (check the file).

## Clustering (as of March 22, 2018)

Implemented tf-idf vectorization of a text (= list of tweets). Then, tried out nearest-neighbors algorithm to find similar tweets (works pretty well). Finally, fed into k-means model
and t-SNE for visualization of the most frequent/important terms.

#### t-SNE results

For the plots, check ```/clustering``` folder with png images.

**Example**

Perplexity = 50, 2D

![perplexity50](clustering/perplexity_50_2.png)

#### References

- [Main article](https://beckernick.github.io/law-clustering/)
- [t-SNE paper](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
- [TF-IDF](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

## Text generation

#### Markov chains results

Example with *n-grams = 3* and *output text size = 45*:

> “every big thinker has had to start as a nobody. just think big  and  iraq encourages distinguishes you from the majority.” – think big if once said he “would be ignoring the law” by granting amnesty through executive action. now he’s about to do it. what ...

## Further analysis

- Have to try out other methods for vectorization (CountVectorizer / Glove).
- Maybe hierarchical clustering instead of k-means? Have to compare those.
- PCA for dimensionality reduction instead of t-SNE.
- Recurrent neural networks with LSTMs for text generation (attention?).

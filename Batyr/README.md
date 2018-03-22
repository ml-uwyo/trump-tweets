# trump-tweets

Repo for Batyr

## Data

`trump_tweets.csv` contains data made by Talon

`trump_tweets_raw.csv` contains data made by me with a help of [Trump Twitter Archive](http://www.trumptwitterarchive.com/archive)

`trump_tweets_cleaned.csv` contains data cleaned up using regex (check the file).

## Clustering (as of March 22, 2018)

Implemented tf-idf vectorization of a text (= list of tweets). Then, tried out nearest-neighbors algorithm to find similar tweets (works pretty well). Finally, fed into k-means model
and t-SNE for visualization of the most frequent/important terms.

#### References

[Main article](https://beckernick.github.io/law-clustering/)
[t-SNE paper](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
[TF-IDF](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

## Further analysis

- Have to try out other methods for vectorization (CountVectorizer / Glove).
- Maybe hierarchical clustering instead of k-means? Have to compare those.
- PCA for dimensionality reduction instead of t-SNE.

# Context 

I was inspired by one of Internet Historian video, [The Engoodening of No Man's Sky](https://www.youtube.com/watch?v=O5BJVO3PDeQ&t=1s), particularly this following segment:

"_Then he starts breaking that (all the feedback of the game from various sources) down into datasets: people who haven't bought the game, people who have bought it and played it for a hundred hours, people who have returned it, etc._

_Then he starts compiling those complaints into usable data, focusing on the people with the most sincere experience of the game._

_Then he starts making a big list of all the things that need adding and prioritize them._"

This workflow was really interesting to me, because it is applicable in many cases, especially in the industry. So here it is, my first attempt on exploring the field of NLP. Enjoy!

# Files
This repo contains this README, a python notebook called Airline Customer Segmentation.ipynb, a slides in pdf format to highlight the most important aspects of this project called 'Airline Customer Segmentation - slides.pdf', and the dataset flight.csv is in the data directory.

# Data
I use the [Steam Reviews Dataset](https://www.kaggle.com/datasets/luthfim/steam-reviews-dataset) from Kaggle, compiled by the user LUTHFI MAHENDRA. Since the original dataset contains ~400000 rows, and that is too many for my poor PC to handle, I decided to only use the subset of it. I randomly selected 20000 reviews from the original dataset, which I provide in the data file. This dataset contains the following columns:

- `date_posted`: The date a review is posted.
- `funny`: How many other player think the review is funny.
- `helpful`: How many other player think the review is helpful.
- `hour_played`: How many hour a reviewer play the game before make a review.
- `recommendation`: Whether the reviewer recommended the game or not.
- `review`: The text of user review.
- `title`: The title of the game that is being reviewed.

The important columns are `review` (feature) and `recommendation` (target), but I also use the other columns for data exploration.

# Goals
 The exact goals of this project are:
- Analyze what words are associated with good reviews (i.e. reviewer recommended the game).
- Predicting whether a reviewer recommended the game or not based on the review.

I also want to explore text processing and Bayesian optimization algorithm.

# Methods

I apply basic text processing to the reviews: removing non-alphabetic characters, removing stop words, stemming, and removing irrelevant words. Then I vectorize the document using two vectorizers: the bag-of-words model and TF-IDF to compare the performance of the two. Then I employ Bayesian search using `hyperopt` to tune the hyperparameters in order to get the most optimal model for a vectorizer+classifier combination. The model performance are evaluated using cross-validation method in order to obtain the distributions of the performance metric. The best model is used to make predictions on the held out test data. Finally, I analyze the features of the best model by looking at the words with largest weights to obtain relevant feedback.

This project requires the standard `numpy`, `pandas`, `matplotlib`, `seaborn`, and `sklearn` packages. 

In addition, some non-standard packages include: `hyperopt`, `lightgbm`, and `eli5`.

# Results
I find that the tuned TF-IDF+Logistic Regression model gives the best performance based on the results from cross-validation. Using this model to the test data, we get an accuracy 94.2% and a ROC AUC score of 0.916, which is much better than the baseline model's (BOW+Naive Bayes) accuracy of 83.6% and ROC AUC score of 0.784. This means that our model will be able to classify the sentiments of larger, unseen data, which we can use to gain more comprehensive feedback on the current games. The most common problems in the games include: modding (players want it back), cheater/hacker, and game crashing. These are valuable input for game developers and companies if they want to increase the players' gaming experience.


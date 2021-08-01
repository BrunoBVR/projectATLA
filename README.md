# Project Avatar: The Last Airbender

In this project I scrape the transcripts from the popular show *Avatar: The Last Airbender* from [the fandom wiki](https://avatar.fandom.com/wiki/Avatar_Wiki). With this data, I do some basic EDA focusing on the character lines. I also included in this project the transcripts from *The Legend of Korra*. As a fun side project I build a classifier to identify lines corresponding to Aang or Korra. With this, we can put up an app to explore the data and let users input some text to predict if they "sound like" Aang or Korra.

## Getting the data

Scraping is done using `BeautifulSoup` and basic `pandas` functionality. The scraping process can be found in *GettingTheData.ipynb* and the final scraped data can be found on two locations:
- Inside the `data` folder as a *csv* file (`ATLA-episodes-scripts.csv`)
- Inside the `data` folder as a pickled pandas dataframe object (`df_atla.data`).

**Note**: This `.data` file contains the unprocessed data as it was scraped. However, some other `.data` files are located inside the `data` folder with *cleaner* versions of the data. For example:
* `df_lines.data` contains only text of character lines (no scene description).
* `episodes_dict.data` is a dictionary with a pandas data frame of all episodes individually. Each key corresponds to a episode number and the values are data frames with all lines.
* `ep_names.data` is a helper data frame with each row corresponding to an episode. The columns are episode name, episode number, book it corresponds to (books are like seasons for the show).

Similar files are found for *The Legend of Korra* (`df_atlk.data`, `df_lines_korra.data`, `korra_episodes_dict.data` and `korra_ep_names.data`).

## EDA

Data exploration is done in the notebooks named *ExploringTheData.ipynb* and *ExploringTheData-Korra.ipynb*. In these, I use `plotly` to get some nice interactive visuals and do all heavy lifting with pandas.

The notebook *GetCharacter.ipynb* is for scraping URL for images of characters and their quote lines from [the fandom wiki](https://avatar.fandom.com/wiki/Avatar_Wiki).

## Modeling

The classifier is built in the `Modeling-AangxKorra.ipynb`. It is a binary classification problem with a somewhat imbalanced set (1796 Aang lines vs. 1370 from Korra). I test some models, two different kinds of text vectorization (Cunt vectorizer and TF-IDF) and two different kinds of over sampling methods (SMOTE and RandomOverSample). I use `cross_validate` to test for multiple metrics on cross validated splits and the best performing model (overall) was a Multinomial Naive Bayes with count vectorization and SMOTE for over sampling. This model was tuned for better hyperparameters and the final model (trained on the full data) was saved as a pickle object under `models/Multinomial_CVect_SMOTE.sav`.

For the remainder of the notebook I test some neural network architectures to try to improve the model with not much success. I tried basic ANN with TF-IDF vectorization, Word Embeddings, and CNN. The best performing architecture was Grid Searched for hyperparameter tuning and the while build can be found under `GridSearch-AangxKorra.py`.

## Dashboarding

I build a dashboard using [plotly dash](https://plotly.com/dash/) to show part of the EDA as an interactive app. The code for the app can be found under the `dash-app` folder and the deployed version can be visited [here](https://atla-app.herokuapp.com/).

## Trying multi-class prediction and failing horribly...

The notebooks `Modeling.ipynb` and `Modeling-script.ipynb` are attempts at predicting lines from the top 10 characters with more lines. It started as a multi-class (10 classes) classification problem but I could not get a good model out. If you want to see me fail horribly at this task, the notebooks are a great place for it! Or, maybe you could have some ideas as to how to make it a reality and that would be awesome!

Stay safe all!

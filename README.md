# 📝 Sentiment Analysis of Reviews

A Natural Language Processing (NLP) project that analyzes customer reviews and classifies them as **Positive** or **Negative**.

The project focuses on preprocessing review text and preparing it for a machine learning sentiment classification model.

## 📌 Project Overview

The goal of this project is to develop a sentiment analysis system that can:

* Analyze customer reviews
* Classify reviews as **Positive** or **Negative**
* Clean and preprocess text data
* Apply different NLP preprocessing techniques
* Prepare text for machine learning
* Predict the sentiment of new/unseen reviews

## 📊 Dataset

The project uses a `Reviews.csv` dataset containing **1,000 customer reviews**.

| Column   | Description          |
| -------- | -------------------- |
| `Review` | Customer review text |
| `Liked`  | Sentiment label      |
| `1`      | Positive review      |
| `0`      | Negative review      |

The dataset contains 1,000 rows and 2 columns, with no missing values.

### Example

```text
Review: Wow... Loved this place.
Liked: 1

Review: Crust is not good.
Liked: 0
```

## 🔄 NLP Preprocessing

The project explores several text preprocessing techniques.

### 1. Lowercasing

Converts all review text into lowercase to maintain consistency.

```python
data['Review'].str.lower()
```

### 2. Tokenization

Breaks sentences into individual words or tokens using NLTK.

```python
from nltk.tokenize import word_tokenize

data['Tokens'] = data['Review'].apply(word_tokenize)
```

### 3. Stop Word Removal

Removes commonly used English words that generally provide less information for the model.

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
```

### 4. Stemming

Reduces words to their root form using `PorterStemmer`.

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
```

For example:

```text
driving → drive
driven  → drive
```

### 5. Lemmatization

Converts words into their meaningful base form using `WordNetLemmatizer`.

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
```

### 6. Removing Numbers

Regular expressions are used to remove numeric characters from reviews.

```python
import re

data['no_numbers'] = data['Review'].apply(
    lambda x: re.sub(r'\d+', ' ', x)
)
```

### 7. Removing Special Characters

Special characters are removed to create cleaner text.

```python
data['cleaned_text'] = data['Review'].apply(
    lambda x: re.sub(r'[^A-Za-z0-9\s]', ' ', x)
)
```

### 8. Emoji Processing

The project also explores converting emojis into text representations using the `emoji` library.

```python
import emoji

data['Emoji'] = data['Review'].apply(emoji.demojize)
```

### 9. HTML Tag Removal

BeautifulSoup is used to remove HTML tags from text.

```python
from bs4 import BeautifulSoup

data['cleaned'] = data['Review'].apply(
    lambda x: BeautifulSoup(x, "html.parser").get_text()
)
```

## 📈 Exploratory Data Analysis

The project includes basic analysis of the review dataset.

### Word Cloud

A word cloud is generated from the combined review text to visualize frequently occurring words.

```python
from wordcloud import WordCloud

combined_text = " ".join(data['Review'])

wordcloud = WordCloud(
    width=800,
    height=300,
    background_color="white"
).generate(combined_text)
```

### Word Frequency Analysis

The project also checks the frequency of selected words such as:

* good
* amazing
* great
* service
* place
* friendly
* love
* food
* really

For example, `food`, `place`, `good`, and `great` appear frequently in the dataset.

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NLTK**
* **Regular Expressions**
* **WordCloud**
* **Matplotlib**
* **Seaborn**
* **BeautifulSoup**
* **Emoji**

## 📁 Project Structure

```text
Sentiment-Analysis/
│
├── Reviews.csv
├── sentiment_analysis.ipynb
├── README.md
└── requirements.txt
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
```

Install the required packages:

```bash
pip install pandas nltk matplotlib seaborn wordcloud beautifulsoup4 emoji
```

Download the required NLTK resources:

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## ▶️ How to Run

1. Clone the repository.
2. Place `Reviews.csv` in the project directory.
3. Install the required dependencies.
4. Open `sentiment_analysis.ipynb`.
5. Run the notebook cells sequentially.
6. Perform text preprocessing and analysis.
7. Use the processed text for sentiment classification.

## 🎯 Project Goal

The main objective is to understand how NLP techniques can be applied to customer reviews and prepare textual data for sentiment classification.

The workflow is:

```text
Customer Reviews
       ↓
Data Loading
       ↓
Data Cleaning
       ↓
Lowercasing
       ↓
Tokenization
       ↓
Stop Word Removal
       ↓
Stemming / Lemmatization
       ↓
Text Cleaning
       ↓
Feature Extraction
       ↓
Sentiment Classification
       ↓
Positive / Negative
```

## 🔮 Future Improvements

* Add TF-IDF or Bag-of-Words feature extraction
* Train machine learning models such as Logistic Regression, Naive Bayes, or SVM
* Compare multiple classification algorithms
* Calculate accuracy, precision, recall, and F1-score
* Add a confusion matrix
* Build a Streamlit web application
* Allow users to enter their own reviews
* Predict sentiment for new/unseen reviews

## 👨‍💻 Author

**Hema Sriram**

B.Tech – Artificial Intelligence & Machine Learning

* GitHub: https://github.com/hemasriram111
* LinkedIn: https://www.linkedin.com/in/hemasriram/
* Portfolio: https://hemasriram111.github.io/portfolio/

---

⭐ If you found this project useful, consider giving the repository a star!

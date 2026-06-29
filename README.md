
# Sentiment Analysis Tool

## Overview
This project implements a Sentiment Analysis tool using Machine Learning. It classifies IMDb movie reviews as **Positive** or **Negative** using the TF-IDF feature extraction technique and the Multinomial Naive Bayes classifier.

## Features
- Load IMDb movie review dataset
- Text preprocessing
  - Lowercasing
  - Remove URLs
  - Remove numbers
  - Remove punctuation
  - Remove extra spaces
- TF-IDF Vectorization
- Multinomial Naive Bayes Classification
- Model Evaluation
  - Accuracy
  - F1 Score
  - Classification Report
- Command Line Interface (CLI) for sentiment prediction

## Technologies Used
- Python
- Pandas
- Scikit-learn

## Dataset
IMDb Movie Reviews Dataset

Columns:
- review
- sentiment

## Installation

```bash
pip install pandas scikit-learn
```

## Run

```bash
python sentiment_analysis.py
```

## Example

Input:

```
This movie was amazing.
```

Output:

```
Predicted Sentiment: positive
```

Input:

```
Worst movie ever.
```

Output:

```
Predicted Sentiment: negative
```

## Project Structure

```
├── IMDB Dataset.csv
├── sentiment_analysis.py



## Author

Your Name

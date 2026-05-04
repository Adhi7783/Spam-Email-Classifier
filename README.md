# Spam Email Classifier

## Overview

This repository houses a lightweight, machine-learning-based spam email classifier. It uses Natural Language Processing (NLP) with `CountVectorizer` and a `LogisticRegression` model to categorize incoming text as either spam or ham. The dataset in `spam_ai.py` is an example placeholder.

## Tech Stack

- Python 3
- NumPy and Pandas for data handling
- Scikit-Learn for vectorization and model training

## Installation

To run this classifier locally, clone the repository and install the required dependencies.

1. Clone the repo:

	```bash
	git clone https://github.com/Adhi7783/Spam-Email-Classifier.git
	cd Spam-Email-Classifier
	```

2. Install dependencies:

	```bash
	pip install numpy pandas scikit-learn
	```

## Usage

Run the main script to train the model on the sample placeholder dataset and print the accuracy, precision, recall, and classification report.

```bash
python spam_ai.py
```

## Project Structure

```text
spam_ai.py
README.md
```
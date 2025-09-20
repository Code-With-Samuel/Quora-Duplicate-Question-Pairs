# Quora Duplicate Question Pairs

## 🚀 Project Overview
This project helps you automatically detect whether two questions asked on Quora are duplicates. It uses powerful machine learning and natural language processing (NLP) techniques to analyze question pairs, extract meaningful features, and predict if they are asking the same thing. This can save time, improve search results, and help organize large Q&A platforms.

## 📦 Dataset Details
- **train.csv**: The main dataset contains thousands of question pairs. Each row has:
  - `qid1`, `qid2`: Unique IDs for each question.
  - `question1`, `question2`: The actual questions in text.
  - `is_duplicate`: 1 if the questions are duplicates, 0 otherwise.
- **How question IDs work**: Each question has a unique ID, but the same question can appear in many pairs. For example, if question ID `2561` appears 88 times, it means that question was compared with 88 other questions.

## 🧩 Main Features & Workflow
1. **Exploratory Data Analysis (EDA)**
	- Understand the data: missing values, duplicate rows, repeated questions.
	- Visualize distributions and relationships using Jupyter notebooks.
2. **Feature Engineering**
	- Extract features like word overlap, token similarity, and bag-of-words (BoW) representations.
	- Use advanced NLP tools to capture the meaning and structure of questions.
3. **Model Training & Evaluation**
	- Train machine learning models (XGBoost, scikit-learn, etc.) to classify question pairs.
	- Evaluate model performance and tune for best results.
4. **Interactive Web App**
	- Use the Streamlit app to enter any two questions and instantly see if they are duplicates.

## 🗂️ Project Structure
- `hello.py`: Simple script to test your environment.
- `streamlit_app/`: Contains the interactive web app and helper functions.
  - `app.py`: Main Streamlit interface for predictions.
  - `helper.py`: Functions for text cleaning, feature extraction, and more.
- Jupyter notebooks for EDA, feature engineering, and model building.
- `requirements.txt` & `pyproject.toml`: All required Python packages listed for easy setup.
- Pre-trained model and vectorizer files (`model.pkl`, `cv.pkl`).

## 🛠️ Installation & Setup
1. **Clone the repository**
	```
	git clone https://github.com/Code-With-Samuel/Quora-Duplicate-Question-Pairs.git
	cd Quora-Duplicate-Question-Pairs
	```
2. **Install dependencies**
	```
	pip install -r requirements.txt
	```
3. **(Optional) Create a virtual environment**
	```
	python -m venv venv
	venv\Scripts\activate  # On Windows
	source venv/bin/activate  # On Mac/Linux
	```

## 💡 How to Use
### 1. Explore the Data
- Open Jupyter notebooks (like `initial_EDA.ipynb`) to understand and visualize the dataset.

### 2. Train & Test Models
- Use the provided notebooks and scripts to train your own models or use the pre-trained ones.

### 3. Try the Web App
- Run the Streamlit app:
	```
	streamlit run streamlit_app/app.py
	```
- Enter any two questions and get an instant prediction: "Duplicate" or "Not Duplicate".

## 🔍 Key Concepts Explained Simply
- **Question IDs**: Each question has a unique number, but popular questions can appear in many pairs. This helps us find which questions are asked most often.
- **Feature Extraction**: The code looks at things like how many words are shared, how similar the sentences are, and other smart ways to compare questions.
- **Value Counts**: Shows how many times each question is used. If a question is repeated a lot, it might be a common or trending topic.

## 📊 Example
Suppose you see this output:
```
2561      88
30782     120
4044      111
...
```
This means question ID `2561` was involved in 88 different pairs. It does NOT mean there are 88 different questions with that ID—just that it was compared with 88 other questions.

## 🧪 Dependencies
Main Python libraries used:
- numpy, pandas, seaborn, matplotlib, scikit-learn, xgboost, distance, fuzzywuzzy, nltk, BeautifulSoup4, plotly, streamlit

## 🌟 Why This Project?
- Helps Quora and similar platforms keep their content clean and organized.
- Saves users time by reducing duplicate answers.
- Demonstrates practical use of NLP and machine learning for real-world problems.

## 📄 License
MIT License (add details if needed)

---
**This project is designed to be beginner-friendly, well-documented, and easy to extend. Whether you want to learn about NLP, build your own duplicate detection system, or just explore a cool dataset, this repository is a great place to start!**
# Quora Duplicate Question Pairs

## Project Overview
This project analyzes and predicts whether two questions from Quora are duplicates of each other. It uses machine learning and natural language processing (NLP) techniques to extract features, train models, and provide predictions. The dataset contains pairs of questions, each with unique question IDs, and a label indicating if they are duplicates.

## Dataset
- **train.csv**: Contains question pairs with columns for question IDs (`qid1`, `qid2`), the questions themselves, and a label (`is_duplicate`).
- Each question ID is unique to a question, but the same question can appear in multiple pairs, so IDs may repeat across rows.

## Main Features
- **Exploratory Data Analysis (EDA)**: Notebooks like `initial_EDA.ipynb` analyze data distribution, missing values, duplicate rows, and repeated questions.
- **Feature Engineering**: Extracts features such as common words, token features, and uses bag-of-words (BoW) and advanced NLP techniques.
- **Model Training**: Uses machine learning models (e.g., XGBoost, scikit-learn) to classify question pairs as duplicate or not.
- **Web App**: A Streamlit app (`streamlit_app/app.py`) allows users to input two questions and get a prediction.

## Project Structure
- `hello.py`: Simple script for testing the environment.
- `streamlit_app/`: Contains the Streamlit web app and helper functions for feature extraction.
  - `app.py`: Main Streamlit interface.
  - `helper.py`: Functions for text processing and feature extraction.
- `requirements.txt` and `pyproject.toml`: List all required Python packages.
- Multiple Jupyter notebooks for EDA and feature engineering.
- Model and vectorizer files (`model.pkl`, `cv.pkl`) for predictions.

## Installation
1. Clone the repository.
2. Install dependencies:
	```
	pip install -r requirements.txt
	```
3. (Optional) Set up a virtual environment for Python 3.10+.

## Usage
### 1. Data Analysis
- Open and run the Jupyter notebooks (`initial_EDA.ipynb`, `bow-with-basic-features.ipynb`, etc.) to explore and preprocess the data.

### 2. Model Training
- Use the provided notebooks and scripts to train models and generate predictions.

### 3. Web App
- Run the Streamlit app:
	```
	streamlit run streamlit_app/app.py
	```
- Enter two questions to check if they are duplicates.

## Key Concepts Explained
- **Question IDs**: Each question has a unique ID, but the same ID can appear in multiple pairs, showing how often a question is reused.
- **Feature Extraction**: Functions in `helper.py` compute word overlap, token features, and other metrics to help the model distinguish duplicates.
- **Value Counts**: Counting how many times each question ID appears helps identify popular or repeated questions.

## Example
If question ID `2561` appears 88 times, it means that question is involved in 88 different pairs, not that there are 88 different questions with that ID.

## Dependencies
Main libraries used:
- numpy, pandas, seaborn, matplotlib, scikit-learn, xgboost, distance, fuzzywuzzy, nltk, BeautifulSoup4, plotly, streamlit

## License
MIT License (add details if needed)

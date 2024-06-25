from docx import Document
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import emoji
import streamlit as st
import random
import time
import cleantext
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Define emoji mappings
emoji_mapping = {
    "positive": "🙂 Positive",
    "neutral": "😐 Neutral",
    "negative": "☹️ Negative"
}


# Function to analyze text sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "positive", emoji_mapping["positive"]
    elif sentiment == 0:
        return "neutral", emoji_mapping["neutral"]
    else:
        return "negative", emoji_mapping["negative"]


# Function to perform text analysis
def analyze_text(rawtext):
    start = time.time()
    summary = ''
    polarity_count = {"positive": 0, "neutral": 0, "negative": 0}

    if rawtext:
        sentiment, emoji = analyze_sentiment(rawtext)
        polarity_count[sentiment] += 1

        blob = TextBlob(rawtext)
        received_text = str(blob)

        words = blob.words
        number_of_tokens = len(words)

        nouns = [word for (word, tag) in blob.tags if tag == 'NN']
        summary = ', '.join(random.sample(nouns, min(len(nouns), 5))) if nouns else 'No nouns found'

    end = time.time()
    final_time = end - start

    return received_text, number_of_tokens, polarity_count, summary, final_time, emoji


# Set up the app title and page layout
st.title("Sentiment Analysis")
st.sidebar.header("Options")
st.sidebar.header("Akash Pandey")

# Add a text input widget to the sidebar
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ', round(blob.sentiment.polarity, 2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity, 2))

    pre = st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                 stopwords=True, lowercase=True, numbers=True, punct=True))

# Main app logic for analyzing text
if st.button("Analyze Text"):
    if text:
        received_text, number_of_tokens, polarity_count, summary, final_time, emoji = analyze_text(text)

    # Display analysis results
    st.subheader("Analysis Results")
    st.write("Text:", received_text, style={"font-size": "20px"})
    st.write("Number of Tokens:", number_of_tokens, style={"font-size": "20px"})
    st.write("Summary:", summary, style={"font-size": "20px"})
    st.write("Time Elapsed:", final_time, style={"font-size": "20px"})

    # Visualize sentiment polarity count
    st.subheader("Sentiment Polarity Count")
    fig, ax = plt.subplots()
    ax.bar(polarity_count.keys(), polarity_count.values())
    st.pyplot(fig)

    # Display emoji for sentiment polarity
    st.subheader("Emoji for Sentiment Polarity")
    st.write("Emoji:--> ", emoji, style={"font-size": "400px"})

# Add a file uploader widget to the sidebar for DOCX and Excel files
uploaded_file = st.sidebar.file_uploader("Upload DOCX or Excel File", type=["docx", "xlsx"])

# Add a file uploader widget for CSV file
csv_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Main app logic for analyzing file
if st.sidebar.button("Analyze File"):
    if uploaded_file:
        if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            docx = Document(uploaded_file)
            received_text = '\n'.join([paragraph.text for paragraph in docx.paragraphs])

        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            df = pd.read_excel(uploaded_file)
            received_text = df.to_string()

        received_text, number_of_tokens, polarity_count, summary, final_time, emoji = analyze_text(received_text)

        # Display analysis results
        st.subheader("Analysis Results")
        st.write("Text:", received_text, style={"font-size": "20px"})
        st.write("Number of Tokens:", number_of_tokens, style={"font-size": "20px"})
        st.write("Summary:", summary, style={"font-size": "20px"})
        st.write("Time Elapsed:", final_time, style={"font-size": "20px"})

        # Visualize sentiment polarity count
        st.subheader("Sentiment Polarity Count")
        fig, ax = plt.subplots()
        ax.bar(polarity_count.keys(), polarity_count.values())
        st.pyplot(fig)

        # Display emoji for sentiment polarity
        st.subheader("Emoji for Sentiment Polarity")
        st.write("Emoji:--> ", emoji, style={"font-size": "400px"})

    elif csv_file:
        try:
            df = pd.read_csv(csv_file, encoding='latin1')
            st.write(df)
            st.write("Summary Statistics:")
            st.write(df.describe())

            st.subheader("Data Visualization with Seaborn")
            sns.pairplot(df)
            plt.tight_layout()
            st.pyplot(plt.gcf())
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

import requests
from newspaper import Article
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re

# Ensure 'punkt' is downloaded
try:
    nltk.download('punkt')
except Exception as e:
    print(f"Error downloading punkt: {e}")

# Step 1: Extracting Financial News using Google Finance (or similar sources)
def fetch_news_content(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text, article.title, article.authors, article.publish_date

# Example of extracting content using newspaper3k (NSE:DLF example replaced with a similar source)
news_sources = [
    "https://www.moneycontrol.com/news/business/stocks/dlf-stock-market-analysis-1234567.html"
]

# Collecting content
news_data = []
for url in news_sources:
    try:
        content, title, authors, publish_date = fetch_news_content(url)
        news_data.append({
            'title': title,
            'authors': authors,
            'publish_date': publish_date,
            'content': content
        })
    except Exception as e:
        print(f"Error fetching article from {url}: {e}")

# Convert to DataFrame
news_df = pd.DataFrame(news_data)
print(news_df.head())

# Step 2: Save to File
news_df.to_csv("F:/articles.csv", index=False)

# Step 3: Scraping Text from a Specific Webpage (The Verge example)
response = requests.get("http://www.theverge.com/2017/4/15/15313968/star-wars-battlefront-2-official-trailer-release-2017-celebration")
soup = BeautifulSoup(response.content, 'html.parser')

# Extract paragraphs
paragraphs = soup.find_all('p')
doc_text = ' '.join([para.get_text() for para in paragraphs])
doc_text = re.sub(r'\n', ' ', doc_text)
print(doc_text)

# Save scraped content to file
with open("F:/theverge_article.txt", "w", encoding='utf-8') as file:
    file.write(doc_text)

# Step 4: Tokenizing Words and Sentences from Local File
with open("F:/articles.csv", "r", encoding='utf-8') as file:
    text = file.read()

# Word Tokenization
words = word_tokenize(text)
print(words[:20])
word_freq = nltk.FreqDist(words)

# Convert frequency distribution to DataFrame
word_freq_df = pd.DataFrame(word_freq.items(), columns=['word', 'count'])
word_freq_df = word_freq_df.sort_values(by='count', ascending=False)
print(word_freq_df.head())

# Save word frequencies to CSV
word_freq_df.to_csv("F:/word_frequencies.csv", index=False)

# Sentence Tokenization
sentences = sent_tokenize(text)
print(f"Number of Sentences: {len(sentences)}")
print(sentences[:5])

# Save sentences to file
with open("F:/tokenized_sentences.txt", "w", encoding='utf-8') as file:
    for sentence in sentences:
        file.write(sentence + '\n')
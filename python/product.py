# Importing necessary libraries
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Define paths
base_path = "C:/Users/fiske/OneDrive/Desktop/COMP8260/GA"
amazon_data_path = f"{base_path}/amazon.csv"
best_buy_data_path = f"{base_path}/Best_buy.csv"

# Load Amazon data
amazon_df = pd.read_csv(amazon_data_path)

# Display the first few rows
print("Amazon Data Preview:")
print(amazon_df.head())

# Check sentiment distribution
print("\nSentiment Distribution:")
print(amazon_df["sentiments"].value_counts())

# Step 1: Create fastText labels and texts
amazon_df['fasttext_label'] = '__label__' + amazon_df['sentiments']
amazon_df['fasttext_text'] = amazon_df['fasttext_label'] + ' ' + amazon_df['cleaned_review']

# Filter out neutral sentiments for binary classification
amazon_df = amazon_df[amazon_df['sentiments'] != 'neutral']
amazon_df = amazon_df[['fasttext_text']]


# Load Best Buy data
best_buy_df = pd.read_csv(best_buy_data_path)

# Select relevant columns and clean data
best_buy_df = best_buy_df[['reviews.rating', 'reviews.text']]
best_buy_df['reviews.rating'] = best_buy_df['reviews.rating'].map({1: 'negative', 2: 'negative', 3: 'negative', 4: 'positive', 5: 'positive'})
best_buy_df.dropna(subset=['reviews.rating', 'reviews.text'], inplace=True)

# Create fastText labels and texts
best_buy_df['fasttext_label'] = '__label__' + best_buy_df['reviews.rating']
best_buy_df['fasttext_text'] = best_buy_df['fasttext_label'] + ' ' + best_buy_df['reviews.text']
best_buy_df = best_buy_df[['fasttext_text']]

# Merge Amazon and Best Buy data
merged_data = pd.concat([amazon_df, best_buy_df], ignore_index=True)

# Text cleaning function
def clean_text(fasttext_text):
    fasttext_text = re.sub(r'[^\w\s]', '', fasttext_text)  # Remove special characters
    fasttext_text = re.sub(r'\s+', ' ', fasttext_text)  # Replace multiple spaces
    return fasttext_text.strip().lower()  # Convert to lowercase

# Apply the cleaning function
merged_data['cleaned_text'] = merged_data['fasttext_text'].apply(clean_text)

# Enhanced cleaning function for NLP
def enhanced_clean_text(text):
    # Extract label and content
    parts = text.split(' ', 1)
    if len(parts) != 2:
        return text
    label, content = parts

    # Clean and process text
    content = re.sub(r'http\S+|www\S+|https\S+', '', content)
    content = re.sub(r'\S+@\S+', '', content)
    content = re.sub(r'[^\w\s]', ' ', content)
    content = re.sub(r'\d+', '', content)

    # Tokenization
    tokens = word_tokenize(content)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    cleaned_text = ' '.join(tokens).strip()
    
    return f"{label} {cleaned_text}"

# Apply enhanced cleaning
merged_data['enhanced_cleaned_text'] = merged_data['cleaned_text'].apply(enhanced_clean_text)

# Handle class imbalance
positive_samples = merged_data[merged_data['enhanced_cleaned_text'].str.contains('__label__positive')]
negative_samples = merged_data[merged_data['enhanced_cleaned_text'].str.contains('__label__negative')]

# Upsample the minority class (negative)
negative_samples_upsampled = resample(negative_samples, replace=True, n_samples=len(positive_samples), random_state=42)
merged_data_balanced = pd.concat([positive_samples, negative_samples_upsampled])

# Train-test split with stratification
train_df, test_df = train_test_split(merged_data_balanced, test_size=0.2, random_state=42, stratify=merged_data_balanced['enhanced_cleaned_text'].str.contains('__label__positive'))

# Save data in fastText format
def save_fasttext_format(df, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for text in df['enhanced_cleaned_text']:
            f.write(f"{text}\n")

# Saving the training and test files
save_fasttext_format(train_df, f"{base_path}/amazon_train_enhanced.txt")
save_fasttext_format(test_df, f"{base_path}/amazon_test_enhanced.txt")

print("Training and test files saved successfully!")
print(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")

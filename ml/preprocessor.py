import re

import nltk
from nltk.corpus import stopwords
import string

# Download stopwords if not already downloaded
nltk.download('stopwords')


def get_preprocessed_sentences(text):
    text = _stripped(text)
    text = text.lower()
    sentences = _split_long_sentences(text)
    return sentences


def _stripped(x):
    return " ".join(str(x).split("\n"))


def _split_long_sentences(sentences, max_length=128):
    # Split pattern for sentence terminators
    punct_split_pattern = re.compile(r'(?<=[.!?])\s+')

    # Split pattern for conjunctions and transitional phrases
    conj_split_pattern = re.compile(r'\b(but|however|therefore|meanwhile|for example|e\.g\.,)\b')

    def find_split_point(chunk, max_length):
        # Find the nearest space before the max_length limit
        split_point = chunk.rfind(' ', 0, max_length)
        return split_point if split_point != -1 else max_length

    def split_and_respect_length(sentence, max_length):
        # First split by punctuation
        primary_chunks = punct_split_pattern.split(sentence)
        new_sentences = []

        for chunk in primary_chunks:
            while len(chunk) > max_length:
                # Further split by conjunctions if the chunk is too long
                secondary_chunks = conj_split_pattern.split(chunk, 1)  # Split at the first occurrence

                if len(secondary_chunks) == 1:
                    # Forced split if no suitable conjunctions found
                    split_point = find_split_point(chunk, max_length)
                else:
                    # Split at the conjunction or nearest space before max_length
                    split_point = find_split_point(secondary_chunks[0], max_length)

                new_sentences.append(chunk[:split_point].strip())
                chunk = chunk[split_point:].strip()

            new_sentences.append(chunk)

        return new_sentences

    new_sentences = []
    for sentence in [sentences]:
        # Assuming remove_stopwords_and_punctuation is a defined function
        sentence = _remove_stopwords_and_punctuation(sentence)
        new_sentences.extend(split_and_respect_length(sentence, max_length))

    return new_sentences


def _remove_stopwords_and_punctuation(text):
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text)

    # Get English stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    # Remove stopwords and punctuation
    cleaned_tokens = [word for word in tokens if word.lower() not in stop_words and word not in punctuation]

    # Join the tokens back into a string
    cleaned_text = ' '.join(cleaned_tokens)

    return cleaned_text

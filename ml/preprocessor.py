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


def _split_long_sentences(sentences, max_length=500):
    new_sentences = []
    for sentence in [sentences]:
        sentence = _remove_stopwords_and_punctuation(sentence)
        if len(sentence) > max_length:
            chunks = [sentence[i:i + max_length] for i in range(0, len(sentence), max_length)]

            new_sentences.extend(chunks)
        elif sentence == " " or sentence == "":
            continue
        else:
            new_sentences.append(sentence)
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

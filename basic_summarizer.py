# first we are going to get a sample text blurb - we'll start off with a hardcoded variable

# now we need to tokenize the input, into sentences and words
# using NLTK - natural language toolkit, python package providing text processing libraries for classification, tokenization, stemming, tagging, parsing, semantic reasoning, etc.
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string

nltk.download('punkt')
nltk.download('stopwords')

# simplistic model of summarization. after removing stopgap words that dont add to the content of the passage, it ranks the words that are use most often, then
# prioritizes sentences that use more of those words. Not generating a new summary, but almost like sorting sentences based off who has highest frequency of words
# throughout the passage.
def summarize_text(text, n=3):
    sentences = sent_tokenize(text) # tokenize into sentences
    words = word_tokenize(text) # tokenize into words
    stop_words = set(stopwords.words('english') + list(string.punctuation) + ['“', "''", "'s", "said", "—", "could", "should", "would", "‘", "’", "”", "also", "``", ":"]) # collect stopwords like the, is, alongside other punctuation
    words = [word for word in words if word.lower() not in stop_words] # remove the stopwords
    word_freq = Counter(words) # count frequency of remaining words

    sentence_scores = {}
    for sentence in sentences: # score each of the sentences based off of the frequency of each word in them
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]
    
    # sort sentences by score and select top N
    summarized = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:n]

    return summarized

print()
print()
print()

# read article from text file.
with open('./article.txt', 'r') as f:
    text = f.read()
    
summary = summarize_text(text, n=3)
print("SUMMARY:")
for sentence in summary:
    print(f"- {sentence}")
print()
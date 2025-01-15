from transformers import pipeline
from newspaper import Article

# pull text off of an article online, from just its URL
def get_article(url):
    article = Article(url)
    
    # download, parse article into text format
    article.download()
    article.parse()
    
    return article.text

# much more efficiant way - we will be using hugging face transformers to use a pre-trained ML pipeline to do the summarization
def summarize_text(text):
    summarizer = pipeline("summarization", model='t5-base') # specify which model we are to use transformers pipelines for
    
    summary = summarizer(text, max_length=150, min_length=80, do_sample=False) # summarize the article using the pre-trained model

    return summary

# read article from text file.
url = 'https://apnews.com/article/hegseth-defense-trump-women-allegations-cd3bfa2dbc35afa37a7f98b72e576233'
text = get_article(url)
summary = summarize_text(text)
print()

print("SUMMARY:")
for sentence in summary[0]['summary_text'].split(' . '):
    print(f"• {sentence[0].upper()}{sentence[1:]}{"." if sentence.strip()[len(sentence.strip())-1] != '.' else ''}") # list out bullet points of the summary, ending each point with a period
print()
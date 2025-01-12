from transformers import pipeline


# much more efficiant way - we will be using hugging face transformers to use a pre-trained ML pipeline to do the summarization
def summarize_text(text):
    summarizer = pipeline("summarization", model='t5-base') # specify which model we are to use transformers pipelines for
    
    summary = summarizer(text, max_length=150, min_length=80, do_sample=False) # summarize the article using the pre-trained model

    return summary

# read article from text file.
with open('./meta_content.txt', 'r') as f:
    text = f.read()
    
summary = summarize_text(text)
    
print()

print("SUMMARY:")
for sentence in summary[0]['summary_text'].split(' . '):
    print(f"• {sentence[0].upper()}{sentence[1:]}{"." if sentence.strip()[len(sentence.strip())-1] != '.' else ''}") # list out bullet points of the summary, ending each point with a period
print()
from transformers import pipeline, T5Tokenizer

def summarize_text(text, max_length=130, min_length=30, model_name='t5-small'):
    """
    Summarizes long texts using a pre-trained model from Hugging Face's Transformers. 
    This enhanced version includes functionality to handle large texts by breaking them 
    into manageable chunks, addressing the token limit constraint of the model.
    
    Parameters:
    - text: The text to be summarized.
    - max_length: The maximum length of the summary (default: 130 words).
    - min_length: The minimum length of the summary (default: 30 words).
    - model_name: The model to be used for summarization. Defaults to 't5-small', 
                  but can be changed to other models like 'facebook/bart-large-cnn' 
                  for potentially better results.
    
    Returns:
    - A string containing the summarized text.
    
    Example usage:
    summary = summarize_text("Your long article or text here...")
    print(summary)
    """
    # Initialize tokenizer and summarizer pipeline with the specified model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model_name)
    
    # Check if the text exceeds the model's maximum token limit
    tokens = tokenizer.tokenize(text)
    max_tokens = tokenizer.model_max_length
    
    if len(tokens) > max_tokens:
        # Break the text into chunks
        chunk_size = max_tokens - 50  # Adjusted for context overlap
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        summaries = []
        for chunk in text_chunks:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        # Combine summaries of each chunk
        return ' '.join(summaries)
    else:
        # Directly summarize if within token limit
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']

# Example usage
if __name__ == "__main__":
    # Example long text input
    text = "Your long article or text here..."
    summary = summarize_text(text)
    print("Summary:")
    print(summary)


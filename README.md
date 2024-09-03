# Text Summarization using Hugging Face Transformers

## Overview

This Python script provides a simple and effective way to summarize long texts using a pre-trained model from Hugging Face's Transformers library. It includes enhanced functionality to handle large texts by automatically breaking them into manageable chunks, addressing the token limit constraint of the model.

## Requirements

- Python 3.6 or higher
- `transformers` library from Hugging Face

To install the `transformers` library, use pip:

```bash
pip install transformers
```

## Functionality

The script includes a function `summarize_text` that takes a long piece of text and summarizes it using a specified pre-trained model. If the text exceeds the model's token limit, the function automatically breaks the text into smaller chunks and summarizes each chunk separately.

### Parameters

- **`text`**: The input text to be summarized.
- **`max_length`**: The maximum length of the summary. Default is `130` words.
- **`min_length`**: The minimum length of the summary. Default is `30` words.
- **`model_name`**: The pre-trained model to be used for summarization. Defaults to `'t5-small'`. This can be changed to other models, such as `'facebook/bart-large-cnn'`, for potentially better results.

### Returns

- A string containing the summarized version of the input text.

## How It Works

1. **Initialization**: The function initializes the tokenizer and summarization pipeline using the specified model (`T5Tokenizer` and `pipeline` from the `transformers` library).
2. **Token Limit Check**: The input text is tokenized to check if it exceeds the model's maximum token limit.
3. **Text Chunking**: If the text exceeds the token limit, it is broken down into smaller chunks that fit within the model's capacity. Each chunk is processed separately.
4. **Summarization**: Each chunk is summarized individually, and the resulting summaries are concatenated to form the final summarized output.
5. **Direct Summarization**: If the text is within the token limit, it is directly summarized without the need for chunking.

## Example Usage

Here's an example of how to use the `summarize_text` function:

```python
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
```

### Running the Script

To run the script, simply copy and paste it into a Python file (e.g., `summarize.py`) and execute it from the command line:

```bash
python summarize.py
```

### Example Output

For an input text like:

```
text = "Artificial Intelligence (AI) is a rapidly growing field of technology..."
```

The output might look like:

```
Summary:
AI is an expanding field focused on creating machines capable of performing tasks that typically require human intelligence...
```

## Customizing the Model

You can change the model used for summarization by specifying a different `model_name`:

```python
summary = summarize_text("Your text here...", model_name='facebook/bart-large-cnn')
```

Different models may provide varying levels of performance and summary quality depending on the context and length of the input text.

## Note

- The function uses the `T5Tokenizer` and Hugging Face's pipeline for summarization.
- Make sure that you have sufficient computational resources, as running large models may require a GPU and significant memory.

## Conclusion

This script offers a flexible and efficient way to summarize long texts using state-of-the-art models from Hugging Face's Transformers library. The ability to handle large texts by breaking them into manageable chunks makes it robust and versatile for various applications, including summarizing articles, reports, or any lengthy documents.

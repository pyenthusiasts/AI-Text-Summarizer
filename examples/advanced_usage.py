"""
Advanced usage example for AI Text Summarizer.

This example demonstrates advanced features like custom configuration,
different models, and batch processing.
"""

from ai_text_summarizer import TextSummarizer, SummarizerConfig


def example_custom_config():
    """Example using custom configuration."""
    print("\n" + "=" * 80)
    print("Example 1: Custom Configuration")
    print("=" * 80)

    text = """
    Machine learning is a subset of artificial intelligence that provides systems
    the ability to automatically learn and improve from experience without being
    explicitly programmed. Machine learning focuses on the development of computer
    programs that can access data and use it to learn for themselves.
    """

    config = SummarizerConfig(
        model_name="t5-small",
        max_length=50,
        min_length=20,
        chunk_overlap=30,
    )

    summarizer = TextSummarizer(config=config, verbose=False)
    summary = summarizer.summarize(text)

    print(f"\nOriginal ({len(text)} chars):")
    print(text.strip())
    print(f"\nSummary ({len(summary)} chars):")
    print(summary)


def example_batch_processing():
    """Example of batch processing multiple texts."""
    print("\n" + "=" * 80)
    print("Example 2: Batch Processing")
    print("=" * 80)

    texts = [
        "Python is a high-level, interpreted programming language known for its simplicity and readability.",
        "JavaScript is a versatile programming language primarily used for web development.",
        "Java is a class-based, object-oriented programming language designed to have few implementation dependencies.",
    ]

    summarizer = TextSummarizer(verbose=False)
    summaries = summarizer.batch_summarize(texts, max_length=30, min_length=10)

    print("\nBatch Summarization Results:")
    for i, (original, summary) in enumerate(zip(texts, summaries), 1):
        print(f"\n--- Text {i} ---")
        print(f"Original: {original}")
        print(f"Summary: {summary}")


def example_different_models():
    """Example using different models."""
    print("\n" + "=" * 80)
    print("Example 3: Different Models")
    print("=" * 80)

    text = """
    Deep learning is part of a broader family of machine learning methods based on
    artificial neural networks with representation learning. Learning can be supervised,
    semi-supervised or unsupervised. Deep learning architectures such as deep neural
    networks, deep belief networks, recurrent neural networks, and convolutional neural
    networks have been applied to fields including computer vision, speech recognition,
    natural language processing, machine translation, bioinformatics, drug design,
    medical image analysis, and board game programs.
    """

    # Note: You can try different models, but they need to be downloaded first
    models = ["t5-small"]  # Add "facebook/bart-large-cnn" if you have it

    for model_name in models:
        print(f"\nUsing model: {model_name}")
        try:
            config = SummarizerConfig(model_name=model_name, max_length=60, min_length=20)
            summarizer = TextSummarizer(config=config, verbose=False)
            summary = summarizer.summarize(text)
            print(f"Summary: {summary}")
        except Exception as e:
            print(f"Error with {model_name}: {e}")


def example_long_text():
    """Example with very long text that requires chunking."""
    print("\n" + "=" * 80)
    print("Example 4: Long Text (Chunking)")
    print("=" * 80)

    # Create a long text by repeating
    short_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science,
    and artificial intelligence concerned with the interactions between computers and
    human language, in particular how to program computers to process and analyze large
    amounts of natural language data. The result is a computer capable of understanding
    the contents of documents, including the contextual nuances of the language within them.
    """

    long_text = short_text * 10  # Repeat to make it long

    config = SummarizerConfig(max_length=150, min_length=50)
    summarizer = TextSummarizer(config=config, verbose=True)

    print(f"\nOriginal text length: {len(long_text)} characters")
    summary = summarizer.summarize(long_text)
    print(f"\nSummary ({len(summary)} chars):")
    print(summary)


def main():
    """Run all examples."""
    print("=" * 80)
    print("AI Text Summarizer - Advanced Usage Examples")
    print("=" * 80)

    try:
        example_custom_config()
        example_batch_processing()
        example_different_models()
        example_long_text()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Some examples require downloading models, which may take time.")

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

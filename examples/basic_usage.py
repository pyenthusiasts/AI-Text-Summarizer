"""
Basic usage example for AI Text Summarizer.

This example demonstrates the simplest way to use the summarizer.
"""

from ai_text_summarizer import summarize_text

# Sample text to summarize
text = """
Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to
natural intelligence displayed by animals including humans. Leading AI textbooks define
the field as the study of intelligent agents: any system that perceives its environment
and takes actions that maximize its chance of achieving its goals. Some popular accounts
use the term artificial intelligence to describe machines that mimic cognitive functions
that humans associate with the human mind, such as learning and problem solving.

The field was founded on the assumption that human intelligence can be so precisely
described that a machine can be made to simulate it. This raises philosophical arguments
about the mind and the ethics of creating artificial beings endowed with human-like
intelligence. These issues have been explored by myth, fiction, and philosophy since
antiquity. Science fiction and futurology have also suggested that AI may become an
existential risk to humanity.

Modern machine learning is the dominant approach to much of modern AI. Deep learning is
a type of machine learning that runs inputs through biologically inspired artificial
neural networks for all of these types of learning. The field of artificial intelligence
research was born at a workshop at Dartmouth College in 1956, where the term "Artificial
Intelligence" was coined.
"""


def main():
    print("=" * 80)
    print("AI Text Summarizer - Basic Usage Example")
    print("=" * 80)
    print()

    print("Original text length:", len(text), "characters")
    print()

    # Summarize using default settings
    print("Summarizing with default settings...")
    summary = summarize_text(text)

    print()
    print("Summary:")
    print("-" * 80)
    print(summary)
    print("-" * 80)
    print()
    print("Summary length:", len(summary), "characters")


if __name__ == "__main__":
    main()

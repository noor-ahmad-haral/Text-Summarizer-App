# Text Summarizer App 📚💻

Welcome to the Text Summarizer App! This simple yet powerful application allows you to generate summaries of articles using different summarization models and evaluate their performance. 🚀

## How to Use

1. **Input an Article**: Enter the article text in the provided text area.
2. **Choose Models**: Select summarization models for Model 1 and Model 2 from the dropdown menus.
3. **Adjust Settings**: Use the slider to set the summarization length according to your preferences.
4. **Summarize**: Click the "Summarize" button to generate summaries using the selected models.
5. **Evaluate**: If you have reference summaries, click the "Evaluate" button to see the evaluation results, including ROUGE scores.
6. **Clear**: To start fresh, click the "Clear" button.

## Models Available 🤖

- Model 1: [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
- Model 2: [Falconsai/text_summarization](https://huggingface.co/Falconsai/text_summarization)

## Features

- Dynamic summarization using state-of-the-art models.
- Evaluation of generated summaries with ROUGE scores.
- Clear and intuitive user interface with Streamlit.

## Installation 🛠️

To run the Text Summarizer App locally, make sure to have the required libraries installed. You can install them using:

```bash
pip install streamlit transformers rouge_score
```

Once installed, simply run the app script:

```bash
streamlit run your_script_name.py
```

Replace `your_script_name.py` with the name of the file containing the provided code.

Feel free to customize the models, appearance, and functionality to suit your needs!

Happy Summarizing! 📖✨
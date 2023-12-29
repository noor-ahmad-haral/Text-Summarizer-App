import streamlit as st
from transformers import pipeline
from rouge_score import rouge_scorer

# Custom session state class
class SessionState:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.evaluation_result = None  # Add this line to store evaluation result

# Function to get or create session state
def get_session_state():
    if 'session_state' not in st.session_state:
        st.session_state.session_state = SessionState(article_input="", clear=False, model1="facebook/bart-large-cnn", model2="Falconsai/text_summarization")
    return st.session_state.session_state

def evaluate_summaries(summary1, summary2):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(summary1, summary2)
    evaluation_result = {
        'ROUGE-1': scores['rouge1'].fmeasure,
        'ROUGE-2': scores['rouge2'].fmeasure,
        'ROUGE-L': scores['rougeL'].fmeasure
    }
    return evaluation_result

def main():
    st.title("Text Summarizer App")
    st.info("Enter an article and click 'Summarize' to generate a summary.")

    # Get or create a session state to store the state persistently
    session_state = get_session_state()

    # Create a text area for article input
    article_input = st.text_area("Enter the article text:", value=session_state.article_input)

    # Summarization length slider
    summary_length = st.slider("Select summarization length:", min_value=30, max_value=200, value=130)

    # Model selection
    selected_model1 = st.selectbox("Select Model 1:", ["facebook/bart-large-cnn", "Falconsai/text_summarization"], index=0)
    selected_model2 = st.selectbox("Select Model 2:", ["facebook/bart-large-cnn", "Falconsai/text_summarization"], index=1)

    if selected_model1 != session_state.model1 or selected_model2 != session_state.model2:
        session_state.model1 = selected_model1
        session_state.model2 = selected_model2

    # Summarization button
    if st.button("Summarize"):
        if article_input:
            # Display loading spinner
            with st.spinner("Summarizing..."):
                # Summarize the article using the selected models
                summarizer1 = pipeline("summarization", model=session_state.model1)
                summarizer2 = pipeline("summarization", model=session_state.model2)

                summary1 = summarizer1(article_input, max_length=summary_length, min_length=30, do_sample=False)
                summary2 = summarizer2(article_input, max_length=summary_length, min_length=30, do_sample=False)

                # Evaluate the summaries and store the result in session state
                evaluation_result = evaluate_summaries(summary1[0]['summary_text'], summary2[0]['summary_text'])
                session_state.evaluation_result = evaluation_result

                # Display original text
                st.subheader("Original Text:")
                st.write(article_input)

                # Display the summaries
                st.subheader("Summary - Model 1:")
                st.write(summary1[0]['summary_text'])

                st.subheader("Summary - Model 2:")
                st.write(summary2[0]['summary_text'])

    # Evaluate button
    if st.button("Evaluate") and session_state.evaluation_result is not None:
        st.subheader("Evaluation Result:")
        for metric, score in session_state.evaluation_result.items():
            st.write(f"{metric}: {score:.4f}")

    # Clear button
    if st.button("Clear"):
        # Clear the text area and set clear flag to True
        session_state.article_input = ""
        session_state.clear = True

    # Reset the clear flag after the next rerun
    if session_state.clear:
        session_state.clear = False
        st.experimental_rerun()

if __name__ == "__main__":
    main()

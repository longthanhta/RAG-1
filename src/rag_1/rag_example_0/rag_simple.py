import os
import google.generativeai as genai
from dotenv import load_dotenv

# Simple document store
DOCUMENTS = [
    "The Pacific Ocean is the largest ocean on Earth.",
    "Dolphins are highly intelligent marine mammals.",
    "Coral reefs are important for marine biodiversity.",
    "Ho Tay is the deepest part of the world's oceans.", "Some fish can glow in the dark due to bioluminescence."
]

def retrieve_relevant_document(query, documents):
    """
    Returns the document with the most keyword overlap with the query.
    """
    query_words = set(query.lower().split())
    best_doc = ""
    max_overlap = 0
    for doc in documents:
        doc_words = set(doc.lower().split())
        overlap = len(query_words & doc_words)
        if overlap > max_overlap:
            max_overlap = overlap
            best_doc = doc
    return best_doc

def simple_rag_demo():
    """
    Simple RAG: retrieve a relevant document and use it to answer a question.
    """
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY_FREE")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY_FREE not found in .env file or environment variables.")

        genai.configure(api_key=api_key)
        model_name = "models/gemini-2.5-flash-preview-05-20"
        model = genai.GenerativeModel(model_name)

        # User question
        question = "What is the deepest part of the ocean?"

        # Retrieve relevant document
        context = retrieve_relevant_document(question, DOCUMENTS)
        print(f"Retrieved context: '{context}'")

        # Compose prompt
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer using the context above."

        # Generate answer
        response = model.generate_content(prompt)
        print("\n--- RAG Response ---")
        print(response.text)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    simple_rag_demo()

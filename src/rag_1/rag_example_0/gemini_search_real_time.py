import os
import google.generativeai as genai
from dotenv import load_dotenv
from duckduckgo_search import DDGS

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API key
# It's common to use "GOOGLE_API_KEY" as the environment variable name
# Make sure your .env file has a line like: GOOGLE_API_KEY="YOUR_API_KEY_HERE"
api_key = os.getenv("GOOGLE_API_KEY_FREE")
if not api_key:
    raise ValueError("GOOGLE_API_KEY_FREE not found in environment variables. Please set it in a .env file.")

genai.configure(api_key=api_key)

# Define Gemini model
# Using a stable model is generally recommended for production.
# "gemini-1.5-flash" or "gemini-1.5-pro" are good choices.
# If you specifically need the preview model, keep it, but be aware of its status.
try:
    model = genai.GenerativeModel("gemini-1.5-flash") # Or "gemini-1.5-pro"
except Exception as e:
    print(f"Could not load specified Gemini model. Error: {e}")
    print("Attempting to list and use the first available model.")
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                model = genai.GenerativeModel(m.name)
                print(f"Using model: {m.name}")
                break
        else:
            raise Exception("No suitable Gemini model found.")
    except Exception as inner_e:
        raise Exception(f"Failed to find any suitable Gemini model. Error: {inner_e}")

def search_federal_register(query, num_results=3):
    """
    Searches the Federal Register using DuckDuckGo Search.

    Args:
        query (str): The search query.
        num_results (int): The maximum number of search results to return.

    Returns:
        list: A list of dictionaries, where each dictionary represents a search result.
              Each dictionary contains 'title', 'href', and 'body' (snippet).
    """
    search_query = f"{query} site:federalregister.gov"
    results = []
    try:
        # Use DDGS().text() for text-based search and iterate through results
        with DDGS() as ddgs:
            for r in ddgs.text(keywords=search_query, max_results=num_results):
                results.append(r)
    except Exception as e:
        print(f"Error during DuckDuckGo search: {e}")
    return results

def get_grounded_answer(query):
    """
    Generates a grounded answer based on Federal Register search results.

    Args:
        query (str): The question to answer.

    Returns:
        tuple: A tuple containing:
               - str: The grounded answer from the Gemini model.
               - list: The search results used as context.
    """
    search_results = search_federal_register(query)

    if not search_results:
        return "Could not find relevant information in the Federal Register.", []

    context = "\n".join([f"Title: {res.get('title', 'N/A')}\nSnippet: {res.get('body', 'N/A')}\nURL: {res.get('href', 'N/A')}\n" for res in search_results])

    prompt = f"Question: {query}\n\nAnswer the question concisely based *only* on the Federal Register information provided below. If the information does not contain the answer, state that you cannot answer based on the provided context.\n\nFederal Register Information:\n{context}\n\nAnswer:"

    try:
        response = model.generate_content(prompt)
        return response.text, search_results
    except Exception as e:
        return f"Error generating answer from Gemini model: {e}", search_results

if __name__ == "__main__":
    query = "Recent tax changes 2025"
    answer, sources = get_grounded_answer(query)

    print("Gemini Grounded Answer (Federal Register):\n", answer)
    print("\nSources:")
    if sources:
        for i, source in enumerate(sources):
            print(f"{i+1}. Title: {source.get('title', 'N/A')}")
            print(f"   Snippet: {source.get('body', 'N/A')}")
            print(f"   URL: {source.get('href', 'N/A')}")
    else:
        print("No sources found.")

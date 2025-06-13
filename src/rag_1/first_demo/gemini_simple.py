import os
import google.generativeai as genai
from dotenv import load_dotenv

def try_google_ai_studio_api_from_env():
    """
    Attempts to use the Google AI Studio API by loading the API key
    from a .env file and using an updated model name.
    """
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY_FREE")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY_FREE not found in .env file or environment variables.")

        genai.configure(api_key=api_key)

        # Choose one of the recommended models from your list:
        model_name = "models/gemini-2.5-flash-preview-05-20"


        print(f"Attempting to use model: '{model_name}'")
        model = genai.GenerativeModel(model_name)

        # Send a prompt to the model
        prompt = "Tell me a short, interesting fact about the ocean."
        print(f"Sending prompt: '{prompt}'")
        response = model.generate_content(prompt)

        # Print the response from the model
        print("\n--- API Response ---")
        print(response.text)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # You've already run list_available_models(), so you can directly call the API function.
    try_google_ai_studio_api_from_env()

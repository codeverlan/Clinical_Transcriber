import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def process_text(prompt, response, temperature):
    # Initialize response_text to an empty string
    response_text = ''

    # Convert response to dictionary and extract transcription text
    if hasattr(response, 'dict'):
        response_dict = response.dict()
        response_text = response_dict.get('text', '') 

    # Get model name from environment variable
    model_name = os.getenv('MODEL_NAME')

    # Generate text using OpenAI
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt + '\n\n' + response_text}
        ],
        temperature=temperature,
        max_tokens=500,
    )

    # Extract generated text from OpenAI response
    generated_text = response['choices'][0]['message']['content'].strip()

    # Return modified text
    return generated_text
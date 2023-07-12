import os
import time
from dotenv import load_dotenv
import openai
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv(".env")

def parse_prompts_file(prompts_file):
    prompts = {}
    default_temperature = float(os.getenv('DEFAULT_TEMP'))
    with open(prompts_file, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Skip blank lines
            if line == "":
                i += 1
                continue
            # Get prompt number from the title
            if ":" in line:
                prompt_number, _ = line.split(":")
                prompt_number = int(prompt_number.strip())
                i += 1
            else:
                print(f"Invalid format at line {i+1}: Expected ':'")
                return
            # Get prompt content (all lines until "Temperature:")
            prompt = ""
            while i < len(lines) and not lines[i].startswith("Temperature:"):
                prompt += lines[i]
                i += 1
            # Get temperature (if specified)
            temperature = default_temperature
            if i < len(lines) and lines[i].startswith("Temperature:"):
                _, temperature = lines[i].split(":")
                temperature = float(temperature.strip())
                i += 1
            # Add prompt to dictionary
            prompts[prompt_number] = (prompt, temperature)
    return prompts


def main():
    # Get environment variables
    openai_api_key = os.getenv('OPENAI_API_KEY')
    output_directory = os.getenv('OUTPUT_DIRECTORY')
    output_separator = os.getenv('OUTPUT_SEPARATOR')
    prompts_file = os.getenv('PROMPTS_FILE')
    audio_directory = os.getenv('AUDIO_DIRECTORY')  # New environment variable for the audio directory

    # Set the OpenAI API key
    openai.api_key = openai_api_key

    # Parse prompts file
    prompts = parse_prompts_file(prompts_file)

    # Get all audio files in the specified directory
    audio_files = [audio_file_name for audio_file_name in os.listdir(audio_directory) if not audio_file_name.endswith('.DS_Store')]

    # Process all audio files in the specified directory with progress bar
    process_files(audio_files, lambda file_name: process_audio_file(file_name, prompts, output_directory, output_separator))


def process_audio_file(audio_file_name, prompts, output_directory, output_separator):
    input_filepath = os.path.join(audio_directory, audio_file_name)
    output_filepath = os.path.join(output_directory, f"{os.path.splitext(audio_file_name)[0]}.txt")

    # Transcribe audio file
    with open(input_filepath, 'rb') as audio_file:
        print(f"Processing file: {input_filepath}")  # Print the file path
        response = openai.Audio.transcribe('whisper-1', audio_file)

    # Get prompt number from filename
    prompt_number = int(os.path.splitext(audio_file_name)[0].split('-')[0])

    # Get prompt and temperature for prompt number
    prompt, temperature = prompts.get(prompt_number, (None, None))
    if prompt is None:
        print(f"No prompt found for number '{prompt_number}'")
        return

    # Process text with OpenAI
    output_text = process_text(prompt, response, temperature)

    # Write output to file
    with open(output_filepath, 'w') as file:
        file.write(f"{response}\n{output_separator}\n{output_text}")


def process_files(file_list, process_function):
    total_files = len(file_list)
    successful_files = 0
    with tqdm(total=total_files, desc="Processing files", ncols=100) as pbar:
        for file_name in file_list:
            try:
                process_function(file_name)
                successful_files += 1
                tqdm.write(f"'{file_name}' successfully processed.")
            except Exception as e:
                tqdm.write(f"Error processing '{file_name}': {str(e)}")
            pbar.update(1)
            # Update progress bar
            time.sleep(0.1)
    print(f"{successful_files} of {total_files} files successfully processed.")
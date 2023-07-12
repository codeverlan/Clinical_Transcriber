from tqdm import tqdm
import time

def process_files(file_list, process_function):
    total_files = len(file_list)
    successful_files = 0

    for file_name in tqdm(file_list, desc="Processing files", ncols=100):
        try:
            process_function(file_name)
            successful_files += 1
            tqdm.write(f"'{file_name}' successfully processed.")
        except Exception as e:
            tqdm.write(f"Error processing '{file_name}': {str(e)}")

        # Update progress bar
        time.sleep(0.1)

    print(f"{successful_files} of {total_files} files successfully processed.")
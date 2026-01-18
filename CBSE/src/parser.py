import pdfplumber
import os

# This looks for the file you just downloaded
file_path = r"C:\Users\User\OneDrive\Documents\ML\CBSE\DATA\science_2024.pdf.pdf"

if os.path.exists(file_path):
    with pdfplumber.open(file_path) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text()
        print("Successfully read the first page!")
        print("-" * 30)
        print(text[:500]) # Prints the first 500 characters
else:
    print(f"Error: I can't find the file at {file_path}. Did you put it in the data folder?")
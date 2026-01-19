import pdfplumber
import os
import re

# This looks for the file you just downloaded
file_path = r"C:\Users\User\OneDrive\Documents\ML\CBSE\DATA\science_2024.pdf.pdf"
def analyze_structure(path):
    all_text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            all_text += page.extract_text() + "\n"
    sections = re.findall(r'(?i)SECTION[\s-]*([A-E])',all_text)
    marks = re.findall(r'\(\d+\)', all_text) 

    print(f"Total Sections Found: {len(sections)}")
    print(f"Sections detected: {sections}")

    return all_text
z = analyze_structure(file_path)
sections_list = re.split(r'(?i)SECTION[\s-]*[A-E]', z)

if len(sections_list) > 6:
    actual_section_a = sections_list[6]
    print("\n--- SECTION A ISOLATED ---")
    print(actual_section_a[:9000].strip())

# 1. Your successful "21-Question" Regex
question_blocks = re.split(r'(\d+)\s+', actual_section_a)

final_dataset = []
temp_storage = {} # We use this to prevent duplicates

for i in range(1, len(question_blocks), 2):
    q_num_raw = question_blocks[i]
    q_text = question_blocks[i+1].strip()

    if q_num_raw.isdigit():
        num = int(q_num_raw)
        
        if 1 <= num <= 20:

            if num not in temp_storage or len(q_text) > len(temp_storage[num]):
                temp_storage[num] = q_text

# 2. Now move them from the storage into your final list
for num in sorted(temp_storage.keys()):
    final_dataset.append({
        "question_no": num,
        "body": temp_storage[num]
    })

final_quiz_data = []

for q in final_dataset:
    body = q["body"]
    parts = re.split(r'\s*[a-d]\)\s+',body)

    question_text = parts[0].strip()

    options = [opt.strip() for opt in parts[1:]]

    final_quiz_data.append({
        "number": q['question_no'],
        "question": question_text,
        "options": options
    })
import json
print(json.dumps(final_quiz_data[0], indent=4))

print(f"SUCCESS: Created a clean database of {len(final_dataset)} questions.")
print(f"Numbers verified: {[q['question_no'] for q in final_dataset]}")
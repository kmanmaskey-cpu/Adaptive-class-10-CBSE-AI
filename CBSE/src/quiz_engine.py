import json
import os
import time
from datetime import datetime
import random

# --- CONFIGURATION ---
DATA_FILE = r"C:\Users\User\OneDrive\Documents\ML\CBSE\science_TEST.json"
HISTORY_FILE = r"C:\Users\User\OneDrive\Documents\ML\CBSE\quiz_history.json"
MASTERY_FILE = r"C:\Users\User\OneDrive\Documents\ML\CBSE\mastery.json"

def load_data(path):
    if not os.path.exists(path): return []
    with open(path, 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return []

def load_mastery():
    if not os.path.exists(MASTERY_FILE): return {}
    with open(MASTERY_FILE, 'r') as f:
        try: return json.load(f)
        except: return {}

def save_mastery(data):
    with open(MASTERY_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def update_weight(question_num, is_correct):
    weights = load_mastery()
    q_key = str(question_num)
    current_weight = weights.get(q_key, 10)
    if is_correct:
        new_weight = max(1, current_weight - 2)
    else:
        new_weight = min(50, current_weight + 7)
    weights[q_key] = new_weight
    save_mastery(weights)

def get_smart_questions(all_questions, num_to_pick=10):
    weights_data = load_mastery()
    w_list = [weights_data.get(str(q['number']), 10) for q in all_questions]
    return random.choices(all_questions, weights=w_list, k=min(num_to_pick, len(all_questions)))

def save_stats(score, total, time_taken, missed_ids):
    history = load_data(HISTORY_FILE)
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "score": score,
        "total": total,
        "percentage": round((score/total)*100, 2) if total > 0 else 0,
        "time_seconds": round(time_taken, 2),
        "avg_speed_per_q": round(time_taken/total, 2) if total > 0 else 0,
        "missed_questions": missed_ids
    }
    history.append(entry)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

def run_quiz(review_filter=None, smart_selection=None):
    if smart_selection:
        questions = smart_selection
        print(f"High-priority items selected.")
    elif review_filter:
        all_q = load_data(DATA_FILE)
        questions = [q for q in all_q if q['number'] in review_filter]
        print(f" Focusing on {len(questions)} items.")
    else:
        questions = load_data(DATA_FILE)
        print(f"FULL QUIZ : {len(questions)} Questions.")

    if not questions:
        print("❌ Error: No questions found.")
        return

    score = 0
    missed_ids = []
    start_time = time.time()

    for q in questions:
        print(f"\nQ{q['number']}: {q['question']}")
        for label, opt in zip(['a', 'b', 'c', 'd'], q['options']):
            print(f"  {label}) {opt}")

        ans = input("\nYour answer: ").lower().strip()

        if ans == q["correct"].lower():
            print("✅ Correct!")
            score += 1
            update_weight(q['number'], True)
        else:
            print(f"❌ Wrong! Answer was {q['correct']}")
            missed_ids.append(q['number'])
            update_weight(q['number'], False)

    total_time = time.time() - start_time
    save_stats(score, len(questions), total_time, missed_ids)
    print(f"\nDONE! Score: {score}/{len(questions)}")

if __name__ == "__main__":
    print("1. Full Quiz\n2. Review Mistakes\n3. Smart Adaptive Mode")
    choice = input("Select (1/2/3): ")
    if choice == '2':
        hist = load_data(HISTORY_FILE)
        if hist: run_quiz(review_filter=hist[-1].get("missed_questions", []))
    elif choice == '3':
        run_quiz(smart_selection=get_smart_questions(load_data(DATA_FILE)))
    else:
        run_quiz()
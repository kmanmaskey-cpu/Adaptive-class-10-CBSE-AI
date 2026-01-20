import json
import os
import time
from datetime import datetime

# --- CONFIGURATION ---
DATA_FILE = r"C:\Users\User\OneDrive\Documents\ML\CBSE\science_TEST.json"
HISTORY_FILE = r"C:\Users\User\OneDrive\Documents\ML\CBSE\quiz_history.json"

def load_data(path):
    if not os.path.exists(path): return []
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except: return []

def save_stats(score, total, time_taken):
    history = load_data(HISTORY_FILE)
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "score": score,
        "total": total,
        "percentage": round((score/total)*100, 2),
        "time_seconds": round(time_taken, 2),
        "avg_speed_per_q": round(time_taken/total, 2)
    }
    history.append(entry)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)
    return history

def run_quiz():
    questions = load_data(DATA_FILE)
    if not questions:
        print("❌ Error: No questions found.")
        return

    score = 0
    total = len(questions)
    
    print("\n" + "="*40)
    print("🚀 CBSE SCIENCE: SPEED & ACCURACY TEST")
    print(f"Goal: {total} Questions | Time Starts Now!")
    print("="*40)

    # --- START TIMER ---
    start_time = time.time()

    for q in questions:
        print(f"\nQ{q['number']}: {q['question']}")
        labels = ['a', 'b', 'c', 'd']
        for label, opt in zip(labels, q['options']):
            print(f"  {label}) {opt}")

        ans = input("\nYour answer: ").lower().strip()

        if "correct" in q:
            if ans == q["correct"].lower():
                print("Correct!")
                score += 1
            else:
                print(f" Wrong! Answer was {q['correct']}")

    # --- END TIMER ---
    end_time = time.time()
    total_time = end_time - start_time

    # --- DISPLAY ANALYTICS ---
    print("\n" + "—"*30)
    print(f"FINISHED!")
    print(f"Score:      {score}/{total} ({round(score/total*100, 1)}%)")
    print(f"Total Time: {round(total_time, 1)} seconds")
    print(f"Avg Speed:  {round(total_time/total, 1)} seconds/question")
    
    # Ivy League Insight: Comparing speed vs accuracy
    if score/total >= 0.9 and total_time/total < 15:
        print("STATUS: ELITE (High Accuracy + High Speed)")
    elif score/total >= 0.9:
        print("STATUS: ACCURATE (Great job, but try to speed up!)")
    else:
        print("STATUS: LEARNING (Focus on concepts first, speed later.)")
    print("—"*32)

    save_stats(score, total, total_time)

if __name__ == "__main__":
    run_quiz()
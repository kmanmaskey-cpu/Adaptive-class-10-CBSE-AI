
import random 
WRONG_PENALTY = 7
RIGHT_REWARD = 2

def calculate_new_weight(current_weight, is_correct):
    if is_correct:
        return max(1, current_weight - RIGHT_REWARD)
    return current_weight + WRONG_PENALTY



def get_smart_questions(all_questions, mastery_data, num_to_pick=5):
    """Ranks questions by weight and picks the top ones for the session."""
    # Create a list of (question, weight) pairs
    weighted_list = []
    
    for q in all_questions:
        q_id = str(q['number'])
        # Get weight from mastery, or default to 10 if new
        weight = mastery_data.get(q_id, 10)
        weighted_list.append((q, weight))
    
    # Sort questions so the highest weights are first
    weighted_list.sort(key=lambda x: x[1], reverse=True)
    
    # Take the top 'num_to_pick' questions
    top_questions = [item[0] for item in weighted_list[:num_to_pick]]
    
    # Shuffle just the selection so the order isn't predictable
    random.shuffle(top_questions)
    return top_questions
def calculate_session_stats(session_results):
    """
    Takes a list of booleans [True, False, True] 
    and returns a summary.
    """
    total = len(session_results)
    correct = sum(session_results)
    percentage = (correct / total) * 100 if total > 0 else 0
    
    return {
        "total_questions": total,
        "correct_answers": correct,
        "score_percentage": round(percentage, 2)
    }
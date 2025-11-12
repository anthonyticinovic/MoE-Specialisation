#!/usr/bin/env python3
"""
Test the improved answer extraction logic on existing Stage 3 outputs.
"""

import json
import sys
import re
from pathlib import Path

# Define the NEW extraction function (copied from updated 02_generate_pope_answers.py)
def extract_yes_no_answer(text: str, question: str = None) -> str:
    """
    Extract yes/no answer from model output.
    Handles both concise answers (Stage 2) and elaborate answers (Stage 3).
    """
    text_lower = text.lower().strip()
    
    # Direct matches (most common for Stage 2)
    if text_lower.startswith('yes'):
        return 'yes'
    if text_lower.startswith('no'):
        return 'no'
    
    # Check first few words
    words = text_lower.split()
    if len(words) > 0:
        first_word = words[0].strip('.,!?')
        if first_word == 'yes':
            return 'yes'
        if first_word == 'no':
            return 'no'
    
    # Strong negative indicators
    strong_negative_phrases = ['there is no', 'there are no', 'there isn\'t', 'there aren\'t', 
                               'not visible', 'cannot see', 'no visible', 'absence of',
                               'no sign of', 'does not show', 'does not feature']
    for phrase in strong_negative_phrases:
        if phrase in text_lower[:80]:
            return 'no'
    
    # Strong affirmative indicators
    strong_affirmative_phrases = ['yes,', 'yes there', 'yes it', 'there is a', 'there are', 
                                  'shows a', 'features a', 'depicts a', 'contains a',
                                  'includes a', 'has a', 'with a', 'shows the', 'features the']
    for phrase in strong_affirmative_phrases:
        if phrase in text_lower[:80]:
            return 'yes'
    
    # For Stage 3: Check if the queried object is actually mentioned
    if question:
        match = re.search(r'is there (?:a |an )?(\w+)', question.lower())
        if match:
            queried_object = match.group(1)
            object_mentioned = queried_object in text_lower[:80]
            
            if object_mentioned:
                # Object is mentioned - check if it's in a descriptive context
                if f'the {queried_object}' in text_lower[:80]:
                    return 'yes'
                if any(p in text_lower[:80] for p in [f'{queried_object} is', f'{queried_object} in',
                                                        f'{queried_object} on', f'{queried_object} at']):
                    return 'yes'
            else:
                # Object NOT mentioned but we have descriptive text
                if len(text_lower) < 20 or text_lower.endswith((',', 'and', 'or', 'with', 'in', 'a')):
                    return 'unclear'
                return 'unclear'
    
    # Fallback: Generic patterns without object mention are unclear
    descriptive_patterns = ['the image features', 'the image shows', 'the image depicts',
                           'the scene features', 'the scene shows']
    for pattern in descriptive_patterns:
        if pattern in text_lower[:50]:
            return 'unclear'
    
    if text_lower.startswith('the ') and len(words) > 2:
        return 'unclear'
    
    return 'unclear'

def test_extraction():
    """Test extraction on actual Stage 3 outputs."""
    
    # Load Stage 3 random answers
    answers_file = "/home/aticinovic/MoE-Specialisation/results/pope_evaluation/stage3_random_answers.json"
    with open(answers_file, 'r') as f:
        answers = json.load(f)
    
    # Re-extract answers
    print("Testing improved extraction on first 50 examples:\n")
    print(f"{'Question':<50} {'GT':<4} {'Old':<8} {'New':<8} {'Raw Output'}")
    print("="*120)
    
    old_unclear = 0
    new_unclear = 0
    changed = 0
    
    for i, item in enumerate(answers[:50]):
        question = item['question']
        gt_answer = item['answer']
        old_predicted = item['predicted_answer']
        raw_output = item['raw_output']
        
        # Re-extract with new logic (pass question for context)
        new_predicted = extract_yes_no_answer(raw_output, question)
        
        if old_predicted == 'unclear':
            old_unclear += 1
        if new_predicted == 'unclear':
            new_unclear += 1
        if old_predicted != new_predicted:
            changed += 1
        
        # Show mismatches or unclear cases
        if new_predicted == 'unclear' or old_predicted != new_predicted:
            print(f"{question[:48]:<50} {gt_answer:<4} {old_predicted:<8} {new_predicted:<8} {raw_output[:50]}")
    
    print(f"\n{'='*120}")
    print(f"Old unclear: {old_unclear}/50 ({100*old_unclear/50:.1f}%)")
    print(f"New unclear: {new_unclear}/50 ({100*new_unclear/50:.1f}%)")
    print(f"Changed predictions: {changed}/50 ({100*changed/50:.1f}%)")
    
    # Full dataset statistics
    print(f"\n{'='*120}")
    print("Full dataset re-extraction:\n")
    
    total = len(answers)
    new_yes = 0
    new_no = 0
    new_unclear_total = 0
    
    for item in answers:
        new_pred = extract_yes_no_answer(item['raw_output'], item['question'])
        if new_pred == 'yes':
            new_yes += 1
        elif new_pred == 'no':
            new_no += 1
        else:
            new_unclear_total += 1
    
    print(f"Total: {total}")
    print(f"Yes: {new_yes} ({100*new_yes/total:.1f}%)")
    print(f"No: {new_no} ({100*new_no/total:.1f}%)")
    print(f"Unclear: {new_unclear_total} ({100*new_unclear_total/total:.1f}%)")
    print(f"\nOriginal unclear rate: 99.8%")
    print(f"New unclear rate: {100*new_unclear_total/total:.1f}%")
    print(f"Improvement: {99.8 - 100*new_unclear_total/total:.1f} percentage points")


if __name__ == "__main__":
    test_extraction()

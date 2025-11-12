"""
Test different prompts for Stage 3 captioning to find the most COCO-like output.
"""

# Candidate prompts ranked by likelihood to produce COCO-style captions
PROMPTS = [
    # Option 1: Short and direct (most COCO-like)
    "Caption:",
    
    # Option 2: Simple question
    "What is this?",
    
    # Option 3: Slightly more specific
    "Briefly describe what you see.",
    
    # Option 4: Current prompt
    "Describe this image in detail.",
    
    # Option 5: Caption instruction
    "Write a caption for this image.",
    
    # Option 6: Very minimal
    "",  # No prompt at all
]

print("=== Prompt Candidates for COCO-Style Captions ===\n")
print("Ranked by expected COCO-likeness:\n")

for i, prompt in enumerate(PROMPTS, 1):
    if prompt:
        print(f"{i}. '{prompt}'")
        print(f"   Expected output style: Direct, concise caption")
    else:
        print(f"{i}. [No prompt]")
        print(f"   Expected output style: May be too generic")
    print()

print("\nRecommendation:")
print("=" * 60)
print("Based on COCO caption characteristics (concise, direct, objective):")
print()
print("🥇 BEST: 'Caption:' or 'Briefly describe what you see.'")
print("   - Short, direct instruction")
print("   - Less likely to produce 'The image features...' meta-language")
print("   - Encourages concise output")
print()
print("🥈 GOOD: 'What is this?' or 'Write a caption for this image.'")
print("   - Natural question format (matches LLaVA training)")
print("   - Might still be verbose")
print()
print("🥉 OK: 'Describe this image in detail.' (current)")
print("   - Works but produces verbose, meta-descriptive language")
print("   - 'in detail' encourages longer captions")
print()
print("❌ AVOID: No prompt")
print("   - Returns to generic 'image in the image' pattern")

import json
import os
import random
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from PIL import Image


class LLaVA_Loader(Dataset):
    """
    DataLoader for LLaVA-Instruct-150K dataset.
    
    Handles multi-turn visual instruction conversations by using ALL Q&A pairs.
    Questions are masked (label=-100), only answers contribute to loss.
    Supports train/val split and subset sampling.
    """
    
    def __init__(
        self,
        annotations_file: str,
        image_dir: str,
        clip_processor,
        tokenizer,
        split: str = 'train',
        subset_fraction: float = 1.0,
        val_fraction: float = 0.2,
        seed: int = 42,
        debug: bool = False,
    ):
        """
        Args:
            annotations_file: Path to llava_instruct_150k.json
            image_dir: Path to image directory (COCO train2017 via symlink)
            clip_processor: CLIP image processor
            tokenizer: Text tokenizer
            split: 'train' or 'val'
            subset_fraction: Fraction of total data to use (0.0 to 1.0)
            val_fraction: Fraction of data reserved for validation (0.0 to 1.0)
            seed: Random seed for reproducibility
            debug: If True, print first 3 samples with decoded tokens and labels
        """
        self.annotations_file = annotations_file
        self.image_dir = image_dir
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.split = split
        self.subset_fraction = subset_fraction
        self.val_fraction = val_fraction
        self.seed = seed
        self.debug = debug
        self.debug_counter = 0  # Track how many samples we've debugged
        
        # Load and process data
        self.data = self._load_data()
        
        if split == 'train':
            print(f"Using {len(self.data)} unique samples for training (LLaVA, ALL QA pairs).")
        else:
            print(f"Using {len(self.data)} unique samples for validation (LLaVA, ALL QA pairs).")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load JSON, create train/val split, apply subset fraction."""
        # Load JSON
        with open(self.annotations_file, 'r') as f:
            all_data = json.load(f)
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        
        # Shuffle data with fixed seed
        shuffled_data = all_data.copy()
        random.shuffle(shuffled_data)
        
        # Split into train/val
        total_samples = len(shuffled_data)
        val_size = int(total_samples * self.val_fraction)
        train_size = total_samples - val_size
        
        if self.split == 'train':
            split_data = shuffled_data[:train_size]
        else:  # val
            split_data = shuffled_data[train_size:]
        
        # Apply subset fraction
        if self.subset_fraction < 1.0:
            subset_size = int(len(split_data) * self.subset_fraction)
            split_data = split_data[:subset_size]
        
        # Filter out samples without conversations or images
        valid_data = []
        for item in split_data:
            if 'conversations' in item and len(item['conversations']) >= 2 and 'image' in item:
                valid_data.append(item)
        
        return valid_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            pixel_values: Processed image tensor
            input_ids: Tokenized text (all QA pairs concatenated)
            attention_mask: Attention mask for text
            labels: Token IDs with questions masked as -100, answers unmasked
        """
        sample = self.data[idx]
        
        # Load and process image
        image_filename = sample['image']
        image_path = os.path.join(self.image_dir, image_filename)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Fallback: create a blank image if file not found
            print(f"Warning: Could not load image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        # Process image with CLIP
        pixel_values = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        # Extract ALL Q&A pairs from conversations
        conversations = sample['conversations']
        
        # Validate: conversations must have even length (question-answer pairs)
        if len(conversations) % 2 != 0:
            # If odd, truncate to last complete Q&A pair
            conversations = conversations[:len(conversations) - 1]
        
        # Skip samples with no valid Q&A pairs
        if len(conversations) < 2:
            # Fallback: return empty sequences (will be filtered in collate_fn if needed)
            print(f"Warning: Sample {idx} has no valid Q&A pairs, skipping.")
            return pixel_values, torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long), \
                   torch.tensor([1], dtype=torch.long), torch.tensor([-100], dtype=torch.long)
        
        # Build combined sequence with proper masking
        # Start with BOS token (included in first question tokenization)
        combined_ids = []
        label_mask = []  # True = compute loss (answer), False = mask (question)
        
        for i in range(0, len(conversations), 2):
            question_text = conversations[i]['value']
            answer_text = conversations[i + 1]['value']
            
            # Remove <image> placeholder from question (only appears in first question)
            question_text = question_text.replace('<image>', '').replace('\n', ' ').strip()
            answer_text = answer_text.strip()
            
            # Tokenize question
            if i == 0:
                # First question: include BOS token
                question_encoding = self.tokenizer(
                    question_text,
                    truncation=False,
                    add_special_tokens=True,  # Includes BOS
                    return_tensors='pt',
                )
            else:
                # Subsequent questions: no BOS (already have it)
                question_encoding = self.tokenizer(
                    question_text,
                    truncation=False,
                    add_special_tokens=False,
                    return_tensors='pt',
                )
            
            question_ids = question_encoding['input_ids'].squeeze(0).tolist()
            
            # Tokenize answer (never add special tokens for answers)
            answer_encoding = self.tokenizer(
                answer_text,
                truncation=False,
                add_special_tokens=False,
                return_tensors='pt',
            )
            answer_ids = answer_encoding['input_ids'].squeeze(0).tolist()
            
            # Add to combined sequence
            combined_ids.extend(question_ids)
            label_mask.extend([False] * len(question_ids))  # Questions masked
            
            combined_ids.extend(answer_ids)
            label_mask.extend([True] * len(answer_ids))  # Answers NOT masked
        
        # Add EOS token at the end
        combined_ids.append(self.tokenizer.eos_token_id)
        label_mask.append(True)  # EOS contributes to loss
        
        # Convert to tensors
        combined_ids = torch.tensor(combined_ids, dtype=torch.long)
        label_mask = torch.tensor(label_mask, dtype=torch.bool)
        
        # CRITICAL: Handle truncation if sequence too long
        max_length = 512
        if len(combined_ids) > max_length:
            # Truncate from the end
            combined_ids = combined_ids[:max_length]
            label_mask = label_mask[:max_length]
            
            # Check if we have at least some answer tokens left
            num_answer_tokens = label_mask.sum().item()
            if num_answer_tokens < 5:
                # Very few answer tokens remain - warn but continue
                if self.debug:
                    print(f"Warning: Sample {idx} truncated to only {num_answer_tokens} answer tokens")
        
        # Create labels: -100 for questions, actual token IDs for answers
        labels = torch.where(
            label_mask,
            combined_ids,           # Answer tokens: keep IDs
            torch.tensor(-100)      # Question tokens: mask with -100
        )
        
        # CRITICAL: Handle padding
        current_length = len(combined_ids)
        padding_length = max_length - current_length
        
        if padding_length > 0:
            # Pad input_ids with pad_token_id
            input_ids = torch.cat([
                combined_ids,
                torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
            
            # Pad labels with -100 (masked)
            labels = torch.cat([
                labels,
                torch.full((padding_length,), -100, dtype=torch.long)
            ])
            
            # Attention mask: 1 for real tokens, 0 for padding
            attention_mask = torch.cat([
                torch.ones(current_length, dtype=torch.long),
                torch.zeros(padding_length, dtype=torch.long)
            ])
        else:
            # No padding needed
            input_ids = combined_ids
            attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        
        # Debug logging for first 3 samples
        if self.debug and self.debug_counter < 3:
            self._debug_print_sample(idx, input_ids, labels, attention_mask, label_mask[:current_length])
            self.debug_counter += 1
        
        # Final dtype validation
        input_ids = input_ids.long()
        labels = labels.long()
        attention_mask = attention_mask.long()
        
        return pixel_values, input_ids, attention_mask, labels
    
    def _debug_print_sample(self, idx, input_ids, labels, attention_mask, label_mask):
        """Print detailed debug information for a sample."""
        print(f"\n{'='*80}")
        print(f"DEBUG SAMPLE {idx}")
        print(f"{'='*80}")
        
        # Decode full sequence
        decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"Full decoded text:\n{decoded_text}\n")
        
        # Show token-by-token breakdown (first 100 tokens only for readability)
        print("Token-by-token breakdown (first 100 tokens):")
        print(f"{'Idx':<5} {'Token':<30} {'InputID':<8} {'Label':<8} {'Mask':<6} {'AttnMask':<8}")
        print("-" * 80)
        
        for i in range(min(100, len(input_ids))):
            token = self.tokenizer.decode([input_ids[i].item()])
            input_id = input_ids[i].item()
            label = labels[i].item()
            is_answer = "ANSWER" if i < len(label_mask) and label_mask[i] else "QUEST"
            attn = attention_mask[i].item()
            
            print(f"{i:<5} {repr(token):<30} {input_id:<8} {label:<8} {is_answer:<6} {attn:<8}")
        
        # Statistics
        num_real_tokens = attention_mask.sum().item()
        num_padding = (attention_mask == 0).sum().item()
        num_answer_tokens = (labels != -100).sum().item()
        num_question_tokens = num_real_tokens - num_answer_tokens
        
        print(f"\nStatistics:")
        print(f"  Total length: {len(input_ids)}")
        print(f"  Real tokens: {num_real_tokens}")
        print(f"  Padding tokens: {num_padding}")
        print(f"  Answer tokens (trainable): {num_answer_tokens}")
        print(f"  Question tokens (masked): {num_question_tokens}")
        print(f"  Answer/Total ratio: {num_answer_tokens/num_real_tokens*100:.1f}%")
        
        # Verify masking correctness
        print(f"\nMasking Verification:")
        all_question_masked = all(labels[i] == -100 for i in range(len(label_mask)) if not label_mask[i])
        all_answer_unmasked = all(labels[i] == input_ids[i] for i in range(len(label_mask)) if label_mask[i])
        all_padding_masked = all(labels[i] == -100 for i in range(num_real_tokens, len(labels)))
        
        print(f"  ✓ All question tokens masked: {all_question_masked}")
        print(f"  ✓ All answer tokens unmasked: {all_answer_unmasked}")
        print(f"  ✓ All padding tokens masked: {all_padding_masked}")
        print(f"{'='*80}\n")

"""
Caption Generation Utilities for Vision-Language Model

Provides robust generation functionality for the custom MoE-based VLM.
Handles proper prompt formatting for LLaVA-style instruction following.

Usage:
    from models.utils.generation import CaptionGenerator

    generator = CaptionGenerator(
        model=model,
        tokenizer=tokenizer,
        vision_tower=vision_tower,
        vl_connector=vl_connector
    )

    caption = generator.generate(
        image_path="path/to/image.jpg",
        prompt="Describe this image in detail.",
        max_new_tokens=256,
        temperature=0.7
    )
"""

from typing import Any

import torch
from PIL import Image


class CaptionGenerator:
    """
    Robust caption generator for vision-language models.

    Handles:
    - Proper prompt formatting (LLaVA instruction style)
    - Vision token preparation
    - Text tokenization
    - Generation with configurable parameters
    - Batch generation support
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        vision_tower: torch.nn.Module,
        vl_connector: torch.nn.Module,
        device: str = "cuda",
        default_prompt: str = "Describe this image in detail.",
    ):
        """
        Initialize caption generator.

        Args:
            model: The language model (Mistral with MoE)
            tokenizer: Text tokenizer
            vision_tower: Vision encoder (CLIP)
            vl_connector: Vision-language connector
            device: Device to run on
            default_prompt: Default instruction if none provided
        """
        self.model = model
        self.tokenizer = tokenizer
        self.vision_tower = vision_tower
        self.vl_connector = vl_connector
        self.device = device
        self.default_prompt = default_prompt

        # Set models to eval mode
        self.model.eval()
        self.vision_tower.eval()
        self.vl_connector.eval()

    def format_llava_prompt(self, instruction: str) -> str:
        """
        Format instruction as LLaVA-style conversation.

        LLaVA format expects:
        <image>
        INSTRUCTION

        Where <image> is a special token indicating where visual features go.

        Args:
            instruction: User instruction/question

        Returns:
            Formatted prompt string
        """
        # LLaVA format: the image token comes first, then the instruction
        # The model expects this format from training
        prompt = f"<image>\n{instruction}"
        return prompt

    def _prepare_vision_tokens(self, image_path: str) -> torch.Tensor:
        """
        Process image and extract vision tokens.

        Args:
            image_path: Path to image file

        Returns:
            Vision tokens tensor [1, num_vision_tokens, hidden_dim]
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")

        # Get vision tower's preprocessor
        processor = self.vision_tower.image_processor

        # Preprocess image
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device)

        # Extract vision features
        with torch.no_grad():
            vision_features = self.vision_tower(pixel_values)

            # Handle different CLIP output formats
            if hasattr(vision_features, "last_hidden_state"):
                vision_features = vision_features.last_hidden_state
            elif isinstance(vision_features, tuple):
                vision_features = vision_features[0]

        # Project to language model space
        with torch.no_grad():
            vision_tokens = self.vl_connector(vision_features)  # [1, num_patches, hidden_dim]

        return vision_tokens

    def _prepare_text_tokens(self, text: str) -> torch.Tensor:
        """
        Tokenize text and get embeddings.

        Args:
            text: Input text

        Returns:
            Text embeddings tensor [1, seq_len, hidden_dim]
        """
        # Tokenize
        input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)[
            "input_ids"
        ].to(self.device)

        # Get embeddings
        with torch.no_grad():
            # Access the model's embedding layer
            model_to_use = self.model.module if hasattr(self.model, "module") else self.model
            embeddings = model_to_use.model.embed_tokens(input_ids)

        return embeddings

    def generate(
        self,
        image_path: str,
        prompt: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Generate caption for an image.

        Args:
            image_path: Path to image file
            prompt: Instruction/question (uses default if None)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            do_sample: Whether to use sampling (False = greedy)
            num_beams: Number of beams for beam search
            repetition_penalty: Penalty for repeating tokens
            length_penalty: Penalty for length (>1 encourages longer)
            early_stopping: Stop when all beams hit EOS
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            verbose: Print generation details

        Returns:
            Dict with keys:
                - 'generated_text': The generated caption
                - 'prompt': The formatted prompt used
                - 'input_ids': Generated token IDs
                - 'num_tokens': Number of tokens generated
        """
        # Use default prompt if none provided
        if prompt is None:
            prompt = self.default_prompt

        # Format prompt
        formatted_prompt = self.format_llava_prompt(prompt)

        if verbose:
            print(f"Formatted prompt: {formatted_prompt}")
            print(f"Generating caption for: {image_path}")

        # Prepare vision tokens
        vision_tokens = self._prepare_vision_tokens(image_path)

        # For LLaVA format, we replace <image> with vision tokens
        # Split the prompt at <image> token
        prompt_parts = formatted_prompt.split("<image>")

        # Tokenize parts
        if prompt_parts[0]:  # Text before <image>
            before_tokens = self._prepare_text_tokens(prompt_parts[0])
        else:
            before_tokens = None

        if prompt_parts[1]:  # Text after <image>
            after_tokens = self._prepare_text_tokens(prompt_parts[1])
        else:
            after_tokens = None

        # Combine: [before_text] + [vision] + [after_text]
        embedding_parts = []
        if before_tokens is not None:
            embedding_parts.append(before_tokens)
        embedding_parts.append(vision_tokens)
        if after_tokens is not None:
            embedding_parts.append(after_tokens)

        combined_embeddings = torch.cat(embedding_parts, dim=1)

        if verbose:
            print(f"Input embedding shape: {combined_embeddings.shape}")

        # Set default token IDs if not provided
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=combined_embeddings,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )

        # Decode generated tokens
        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if verbose:
            print(f"Generated {len(generated_ids)} tokens")
            print(f"Caption: {generated_text}")

        return {
            "generated_text": generated_text,
            "prompt": formatted_prompt,
            "input_ids": generated_ids.cpu(),
            "num_tokens": len(generated_ids),
        }

    def generate_batch(
        self, image_paths: list[str], prompts: list[str] | None = None, **generation_kwargs
    ) -> list[dict[str, Any]]:
        """
        Generate captions for multiple images.

        Args:
            image_paths: List of image paths
            prompts: List of prompts (one per image, or None for default)
            **generation_kwargs: Generation parameters (passed to generate())

        Returns:
            List of generation result dictionaries
        """
        results = []

        if prompts is None:
            prompts = [None] * len(image_paths)

        for image_path, prompt in zip(image_paths, prompts, strict=False):
            try:
                result = self.generate(image_path=image_path, prompt=prompt, **generation_kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error generating caption for {image_path}: {e}")
                results.append(
                    {
                        "generated_text": f"[ERROR: {str(e)}]",
                        "prompt": prompt or self.default_prompt,
                        "input_ids": None,
                        "num_tokens": 0,
                        "error": str(e),
                    }
                )

        return results

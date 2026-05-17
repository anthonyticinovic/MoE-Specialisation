"""Synthetic concept-image generator shared by analysis scripts."""

from __future__ import annotations

from PIL import Image, ImageDraw


class SyntheticImageGenerator:
    """Generates simple synthetic images for concept testing."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.colors = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (255, 255, 0),
            "orange": (255, 165, 0),
            "purple": (128, 0, 128),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
        }

    def generate_concept_image(self, concept: str) -> Image.Image:
        """Generate image from concept: 'red', 'circle', or 'red circle'."""
        parts = concept.lower().split()

        # Pure color patch
        if len(parts) == 1 and parts[0] in self.colors:
            return Image.new("RGB", (self.image_size, self.image_size), self.colors[parts[0]])

        # Shape (with optional color)
        if len(parts) == 2:
            # Colored shape: "red circle"
            color_name = parts[0]
            shape_name = parts[1]
            if color_name not in self.colors or shape_name not in ["circle", "square", "triangle"]:
                raise ValueError(f"Unknown concept: {concept}")
            fill_color = self.colors[color_name]
            outline_color = self.colors[color_name]
        elif len(parts) == 1 and parts[0] in ["circle", "square", "triangle"]:
            # Pure shape: black outline only, white fill (no color contamination)
            shape_name = parts[0]
            fill_color = (255, 255, 255)  # white fill
            outline_color = (0, 0, 0)  # black outline
        else:
            raise ValueError(f"Unknown concept: {concept}")

        # Create white background and draw shape
        image = Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        margin = self.image_size // 4
        line_width = 3  # Make outline visible

        if shape_name == "circle":
            draw.ellipse(
                [margin, margin, self.image_size - margin, self.image_size - margin],
                fill=fill_color,
                outline=outline_color,
                width=line_width,
            )
        elif shape_name == "square":
            draw.rectangle(
                [margin, margin, self.image_size - margin, self.image_size - margin],
                fill=fill_color,
                outline=outline_color,
                width=line_width,
            )
        elif shape_name == "triangle":
            center_x = self.image_size // 2
            draw.polygon(
                [
                    (center_x, margin),
                    (margin, self.image_size - margin),
                    (self.image_size - margin, self.image_size - margin),
                ],
                fill=fill_color,
                outline=outline_color,
            )

        return image

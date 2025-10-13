from analysis_scripts.cross_modality_purity import SyntheticImageGenerator
import matplotlib.pyplot as plt
import os

os.makedirs('results', exist_ok=True)
gen = SyntheticImageGenerator()
concepts = ['red', 'blue', 'circle', 'triangle', 'red circle', 'blue triangle']

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, concept in zip(axes.flat, concepts):
    img = gen.generate_concept_image(concept)
    ax.imshow(img)
    ax.set_title(concept, fontsize=14, fontweight='bold')
    ax.axis('off')
plt.tight_layout()
plt.savefig('results/synthetic_images_preview.png', dpi=150, bbox_inches='tight')
print('✅ Saved to: results/synthetic_images_preview.png')

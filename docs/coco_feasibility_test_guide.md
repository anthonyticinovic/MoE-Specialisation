# Comprehensive COCO Feasibility Test

## Overview

This test evaluates 100+ compositional concepts across 13 different case study categories to determine which concepts have sufficient COCO samples (N≥50 unique images) for robust representation analysis.

## Case Study Categories

### 1. **Color Compositional** (Color × Object)
Tests whether the model learns color as an independent feature that can bind to different objects.

**Concepts tested:**
- Apples: red_apple, green_apple, yellow_apple
- Roses: red_rose, pink_rose, white_rose
- Cars: red_car, blue_car, white_car, black_car
- Umbrellas: red_umbrella, blue_umbrella, yellow_umbrella

**Research question:** Does Stage 3 show better color-object binding than Stage 2?

---

### 2. **Size Compositional** (Size × Object)
Tests whether the model learns size as an independent feature.

**Concepts tested:**
- Fruits: small_apple, large_apple
- Animals: small_dog, large_dog
- Objects: small_boat, large_boat

**Research question:** Can the model disentangle size from object identity?

---

### 3. **Base Fruits** (High-frequency single concepts)
Tests basic object recognition with natural variation.

**Concepts tested:** banana, orange, strawberry, blueberry, raspberry, lemon, lime, cherry, pear

**Research question:** What's the baseline cross-modal similarity for common objects?

---

### 4. **Base Vegetables**
Similar to fruits but different semantic category.

**Concepts tested:** tomato, carrot, broccoli, cucumber

---

### 5. **Animals** (Species variety)
Tests animal category representations.

**Concepts tested:** cat, dog, horse, cow, sheep, elephant, giraffe, zebra, bird, duck, seagull

**Research question:** Does the model learn hierarchical categories (animal > mammal > dog)?

---

### 6. **Sports Objects** (Round objects with size variation)
Tests shape consistency across different sports equipment.

**Concepts tested:** soccer_ball, tennis_ball, baseball, basketball, football

**Research question:** Does "round" emerge as an independent visual feature?

---

### 7. **Furniture** (Indoor scene objects)
Tests common indoor objects.

**Concepts tested:** chair, table, couch, bed

---

### 8. **Vehicles** (Transportation category)
Tests vehicle category with different subtypes.

**Concepts tested:** car, truck, bus, train, bicycle, motorcycle

**Research question:** Does Stage 3 learn vehicle as a superordinate category?

---

### 9. **Activity/Actions** (Abstract concepts)
Tests whether the model can capture dynamic actions.

**Concepts tested:** person_sitting, person_standing, person_walking, person_running

**Research question:** Can compositional binding work for person × action?

---

### 10. **States** (Object states)
Tests state representations.

**Concepts tested:** open_door, closed_door, sunny_day, cloudy_sky

---

### 11. **Spatial** (Scene-level concepts)
Tests abstract scene categorization.

**Concepts tested:** indoor_scene, outdoor_scene

---

### 12. **Material/Texture** (Surface properties)
Tests material as a compositional feature.

**Concepts tested:** wooden_table, glass_vase, metal_fork

**Research question:** Does the model learn material as independent from object?

---

### 13. **Prepared Food** (Complex food items)
Tests complex composite objects.

**Concepts tested:** pizza, sandwich, hot_dog, cake, donut

---

## Running the Test

### Quick Run:
```bash
cd /home/aticinovic/MoE-Specialisation
sbatch hpc/analysis_scripts/coco_feasibility_comprehensive.sbatch
```

### Check Results:
```bash
lastoutf  # Shows most recent output file
```

## Output Structure

The test will provide:

1. **Detailed counts** for each concept (matches + unique images)
2. **Feasibility assessment** (✅ YES if ≥50 unique images)
3. **Sample captions** (first 5 per concept for inspection)
4. **Case study summaries** (which studies are feasible)
5. **Overall recommendations** (best case studies to run)
6. **Top 10 most abundant concepts** (ranked by sample count)

## Interpreting Results

### Feasibility Tiers:
- **✅ ≥50 unique images**: Fully feasible for N=50 sampling
- **⚠️ 25-49 images**: Can work with N=25 or relaxed exclusion criteria
- **❌ <25 images**: Need alternative concepts or manual curation

### Sample Ambiguity:
The test includes exclusion criteria to avoid ambiguous samples:
- "red_apple" excludes images also containing "green", "yellow"
- "baseball" excludes "player", "field", "game" to focus on the ball itself
- "orange" excludes "juice" to avoid processed forms

This ensures clean, unambiguous training samples.

## Next Steps After Results

1. **Review case study summaries** to see which categories are most feasible
2. **Check sample captions** to ensure quality matches your intent
3. **Select 4-8 concepts** for your compositional case study
4. **Update config file** (`configs/compositional_coco.json`)
5. **Run full analysis** with unified sbatch script

## Expected Runtime

- ~5-10 minutes on physical partition
- Processes ~600K COCO captions
- Pure CPU task (no GPU needed)

## Design Philosophy

This comprehensive test allows you to:
- **Explore** what compositional structures are available in COCO
- **Validate** whether your theoretical concepts have empirical support
- **Compare** multiple case studies without committing to expensive GPU runs
- **Discover** unexpected high-quality concepts you hadn't considered

The goal is to make case study design data-driven rather than speculative.

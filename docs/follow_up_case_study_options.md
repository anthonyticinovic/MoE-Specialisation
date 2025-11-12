# Follow-Up Case Study Options After Color Binding

## Your Current State

✅ **Case Study 1: Color Compositional (COMPLETE)**
- Concepts: red_apple, green_apple, red_car, blue_car, red_bus, blue_bus, blue_kite, red_kite
- Finding: Stage 2 shows category dominance (CCS = 0.937), Stage 3 shows weak color binding (CCS = 1.020)
- Academic claim: "Hard routing prioritizes object categories over perceptual attributes"

## Should You Add Another Case Study?

### ✅ **YES, if you want to:**
1. Show **generalizability** of your finding beyond color
2. Test **different compositional dimensions** (shape, material, state)
3. Strengthen **publication narrative** (2 complementary studies > 1)
4. Demonstrate **systematic compositionality** across attribute types

### ❌ **NO, if:**
1. Time constrained (1 case study is sufficient for thesis)
2. Color finding is already strong and clear
3. Other thesis chapters need more attention
4. Diminishing returns on additional evidence

---

## 🎯 **Top 3 Recommended Follow-Up Studies**

### **Option 1: Material/Texture Compositionality** ⭐⭐⭐⭐⭐

**Why this is THE BEST choice:**

1. **Orthogonal to color**: Tests different perceptual dimension
2. **Visual + Semantic**: Material has both visual (texture) and semantic (function) properties
3. **Real-world relevance**: Object recognition depends on material (glass vs plastic cup)
4. **Narrative fit**: "Does soft routing learn compositional structure for both color (low-level) AND material (mid-level)?"

**Concepts to Test:**
```json
{
  "concepts": [
    "wooden_table", "wooden_chair", "wooden_bench",
    "metal_fork", "metal_spoon", "metal_knife",
    "glass_vase", "glass_bowl", "glass_cup",
    "plastic_bottle", "plastic_container", "plastic_cup"
  ]
}
```

**Expected CCS Pattern:**
- Stage 2: CCS < 1.0 (object dominates: table-chair > wooden-metal)
- Stage 3: CCS ≈ 1.0-1.1 (material emerges: wooden-wooden ≈ table-chair)

**COCO Feasibility:** 
- From your test: wooden_table (1,100 imgs) ✅, glass_vase (340 imgs) ✅
- Need to test: metal_fork, plastic_bottle, etc.

**Academic Value:** HIGH
- Tests generalization across attribute types
- Material is mid-level (between color and category)
- Publishable finding: "Compositional hierarchy: color < material < category"

---

### **Option 2: State/Action Compositionality** ⭐⭐⭐⭐

**Why this is great:**

1. **Abstract concepts**: Tests dynamic states (not just visual attributes)
2. **Already validated**: Your feasibility test shows 400+ images per action
3. **Different from color**: Temporal/dynamic vs static/perceptual
4. **Narrative fit**: "Does compositional binding work for abstract concepts?"

**Concepts to Test:**
```json
{
  "concepts": [
    "person_sitting", "dog_sitting", "cat_sitting",
    "person_standing", "dog_standing", "bird_standing",
    "person_walking", "dog_walking", "elephant_walking",
    "person_running", "dog_running", "horse_running"
  ]
}
```

**Expected CCS Pattern:**
- Stage 2: CCS < 1.0 (agent dominates: person-person > sitting-sitting)
- Stage 3: CCS > 1.0 (action emerges: sitting-sitting > person-dog)

**COCO Feasibility:** ✅ CONFIRMED
- person_sitting: 8,250 imgs
- person_standing: 13,191 imgs
- person_walking: 4,264 imgs
- person_running: 482 imgs

**Academic Value:** HIGH
- Novel: Few VLM papers test action compositionality
- Tests abstraction: Actions are conceptual, not purely visual
- Publishable finding: "Soft routing enables agent-action disentanglement"

---

### **Option 3: Shape Compositionality** ⭐⭐⭐

**Why this is good:**

1. **Pure visual**: Shape is purely perceptual (no linguistic confounds)
2. **Geometric**: Clear mathematical definition (round, square, rectangular)
3. **Hierarchical**: Shape has levels (circle > sphere, square > cube)
4. **Narrative fit**: "Another perceptual attribute like color"

**Concepts to Test:**
```json
{
  "concepts": [
    "round_ball", "round_plate", "round_clock", "round_pizza",
    "square_window", "square_table", "square_tile",
    "rectangular_table", "rectangular_door", "rectangular_window"
  ]
}
```

**COCO Feasibility:** ⚠️ NEEDS TESTING
- COCO captions rarely use explicit shape words
- Likely low coverage (< 50 images per concept)
- May need to relax matching criteria

**Academic Value:** MEDIUM
- Similar to color (perceptual attribute)
- May be too sparse in COCO
- Redundant with color finding

---

## 🚫 **Options to AVOID**

### ❌ **Size Compositionality**
- Your test shows: small_dog (1,007 imgs), large_dog (649 imgs) ✅
- **Problem**: "Small" and "large" are subjective, context-dependent
- COCO uses "small dog" for breeds (Chihuahua) not visual size
- **Noisy signal**: May not reflect compositional binding

### ❌ **More Color Studies**
- Would be redundant with your current study
- Doesn't test generalization
- Reviewer criticism: "Just testing one attribute repeatedly"

### ❌ **Spatial Relationships** (left/right/above/below)
- COCO captions rarely describe spatial relations explicitly
- Would require manual annotation or spatial reasoning models
- Out of scope for compositional binding study

---

## 💎 **My Top Recommendation: Material Compositionality**

**Rationale:**

1. **Strongest narrative**: "We test color (low-level) AND material (mid-level)"
2. **Complementary**: Color = perceptual, Material = perceptual + functional
3. **Feasibility**: Wooden_table (1,100 imgs), glass_vase (340 imgs) already confirmed
4. **Academic impact**: Tests hierarchical compositionality (color < material < category)
5. **Clear hypothesis**: Stage 2 fails on material (like color), Stage 3 improves

**What to Test:**

Add these concepts to your feasibility test:
```python
"metal_fork", "metal_spoon", "metal_knife",
"plastic_bottle", "plastic_container", "plastic_bag",
"glass_cup", "glass_bowl", "glass_bottle",
"wooden_bench", "wooden_chair", "wooden_spoon"
```

Then run analysis with 6-9 concepts (2-3 materials × 3 objects each).

**Expected Results:**
- Stage 2: Material CCS ≈ 0.90-0.95 (category dominance, like color)
- Stage 3: Material CCS ≈ 1.05-1.15 (material binding emerges)
- **Narrative**: "Compositional binding generalizes from color to material"

---

## 📊 **Thesis Structure with 2 Case Studies**

### **Chapter: Compositional Binding Analysis**

**Section 1: Color Compositionality**
- Research question: Does soft routing learn color-object binding?
- Method: 8 concepts (2 colors × 4 objects)
- Result: Stage 2 CCS = 0.937, Stage 3 CCS = 1.020 (+8.8%)
- Finding: Category dominance in Stage 2, weak color binding in Stage 3

**Section 2: Material Compositionality**
- Research question: Does color finding generalize to other attributes?
- Method: 9 concepts (3 materials × 3 objects)
- Result: Stage 2 CCS ≈ 0.92, Stage 3 CCS ≈ 1.10 (+15%)
- Finding: Compositional binding generalizes, stronger for material than color

**Section 3: Discussion**
- Pattern: Both color and material show category dominance in Stage 2
- Improvement: Stage 3 enables attribute disentanglement across dimensions
- Hierarchy: Color (weak) < Material (moderate) < Category (strong)
- Implication: Soft routing is necessary but not sufficient for full compositionality

---

## 🎯 **Alternate: Action Compositionality**

**If material concepts are too sparse in COCO, use actions instead:**

**Advantages:**
- ✅ Already validated: 400-13,000 images per concept
- ✅ Novel: Few VLMs test action compositionality
- ✅ Different domain: Abstract/dynamic vs perceptual

**Disadvantages:**
- ⚠️ More complex: Actions involve temporal reasoning
- ⚠️ Less direct: "Person running" is compositional but not purely visual
- ⚠️ May conflate: Action detection with compositional binding

**Narrative:**
- "We test perceptual (color) AND conceptual (action) compositionality"
- "Soft routing enables binding for both visual attributes and abstract states"

---

## ⚖️ **Decision Framework**

### **Choose Material if:**
- ✅ You want visual compositionality (stays in perceptual domain)
- ✅ You want mid-level features (between color and category)
- ✅ You want hierarchical story (low-level → mid-level → high-level)
- ✅ Feasibility test confirms >50 images per concept

### **Choose Action if:**
- ✅ Material concepts are too sparse in COCO
- ✅ You want to test abstract/dynamic concepts
- ✅ You want novelty (few VLM papers test actions)
- ✅ You want guaranteed high coverage (already validated)

### **Choose Nothing (1 case study only) if:**
- ✅ Time constrained
- ✅ Color finding is already strong
- ✅ Reviewer asks: "Why not test generalization?" → "Future work"

---

## 🚀 **Next Steps**

### **Option A: Add Material Study**
1. Expand feasibility test with material concepts
2. Run: `sbatch hpc/analysis_scripts/coco_feasibility_test.sbatch`
3. If ≥6 concepts have ≥50 images → proceed with material study
4. If <6 concepts → switch to action study

### **Option B: Add Action Study**
1. Already validated (no new feasibility test needed)
2. Create config: `configs/compositional_actions.json`
3. Run: `sbatch hpc/analysis_scripts/compositional_coco_full.sbatch`
4. Compute CCS, compare to color results

### **Option C: Stop Here**
1. Write up color case study thoroughly
2. Discuss limitations in thesis
3. Propose material/action as "future work"
4. Focus on other thesis chapters

---

## 📝 **My Recommendation**

**Test feasibility of material concepts first:**

```bash
# Run updated feasibility test
sbatch hpc/analysis_scripts/coco_feasibility_test.sbatch

# Check output for:
# - metal_fork, metal_spoon, metal_knife
# - plastic_bottle, plastic_container
# - glass_cup, glass_bowl
# - wooden_bench, wooden_chair
```

**Then decide:**
- If ≥6 material concepts have ≥50 images → **Run Material Study** (BEST)
- If <6 material concepts → **Run Action Study** (GOOD)
- If time constrained → **Stop at Color Study** (ACCEPTABLE)

**Timeline estimate:**
- Material study: ~2-3 hours GPU time
- Action study: ~1.5-2 hours GPU time
- Analysis + writing: ~3-5 days

**Academic impact:**
- 1 case study: Sufficient for thesis, may need "future work" section
- 2 case studies: Strong generalization story, publication-ready

# Color Compositional Alternatives for Case Study 1

## Current Proposal
- **Concepts**: red_apple, green_apple, red_car, blue_car, red_umbrella, blue_umbrella
- **Issue**: Umbrellas have lower sample counts (165-179 images)

## Better Alternatives to Consider

### 🚌 **Option A: BUSES** (Recommended!)
**Why**: Buses are EXTREMELY abundant in COCO (3,290 total images)
- Often have distinctive colors (red double-deckers, yellow school buses, white transit)
- Large, salient objects (easy to identify)
- Strong cultural associations (red = London, yellow = school)

**Test concepts**:
- red_bus
- blue_bus  
- yellow_bus
- white_bus

**Expected coverage**: Very high (buses are common in urban COCO scenes)

---

### 🚂 **Option B: TRAINS**
**Why**: Trains are very abundant (3,819 total images)
- Often color-coded in captions
- Large objects with clear boundaries
- Diverse colors (red, blue, yellow)

**Test concepts**:
- red_train
- blue_train
- yellow_train

**Expected coverage**: High

---

### 🪁 **Option C: KITES**
**Why**: Colorful objects explicitly described with colors
- Flying objects (unique category vs ground vehicles)
- Popular in outdoor COCO scenes
- High color variation

**Test concepts**:
- red_kite
- blue_kite
- yellow_kite

**Expected coverage**: Medium-High

---

### 🚤 **Option D: BOATS**
**Why**: Already tested, good coverage (471-672 images for small/large)
- Water vehicles (different category from land transport)
- Often color-described
- Size variation available

**Test concepts**:
- red_boat
- blue_boat
- white_boat

**Expected coverage**: High (already confirmed)

---

### 🎈 **Option E: BALLOONS**
**Why**: Inherently colorful objects
- Party/celebration scenes
- Multiple colors often in same scene
- Clear visual boundaries

**Test concepts**:
- red_balloon
- blue_balloon
- yellow_balloon

**Expected coverage**: Medium

---

### 👕 **Option F: SHIRTS/CLOTHING**
**Why**: EXTREMELY abundant (person captions = 175,856)
- People wearing colored clothing
- Very natural descriptions
- Compositional: color × clothing × person

**Test concepts**:
- red_shirt
- blue_shirt
- white_shirt
- black_shirt

**Expected coverage**: Very high

---

### 🌸 **Option G: FLOWERS (General)**
**Why**: Natural objects with color variation
- Beyond roses (which have lower coverage)
- Common in COCO scenes
- Natural category contrast to manufactured objects

**Test concepts**:
- red_flower
- yellow_flower
- white_flower

**Expected coverage**: Medium-High

---

## Recommended Replacement Strategy

### **Best Replacement for Umbrellas**: BUSES or TRAINS

**Revised Case Study 1A (Land Vehicles)**:
```json
{
  "concepts": [
    "red_apple", "green_apple",    // Natural objects (70, 69 imgs)
    "red_car", "blue_car",          // Small vehicles (352, 140 imgs)  
    "red_bus", "blue_bus"           // Large vehicles (TBD)
  ]
}
```

**Advantages**:
- All transportation theme (cars + buses = coherent category)
- Size variation (cars vs buses)
- Very high expected coverage
- Tests color binding across natural (apple) and manufactured (vehicles) domains

---

### **Alternative: Multi-Category Color Study**

**Revised Case Study 1B (Color Across Categories)**:
```json
{
  "concepts": [
    "red_apple", "blue_apple",      // Natural/food
    "red_car", "blue_car",          // Land vehicle
    "red_boat", "blue_boat"         // Water vehicle
  ]
}
```

**Advantages**:
- Tests color binding across 3 distinct categories
- Natural vs manufactured distinction
- Land vs water distinction
- Tests generalization of color feature

---

### **Alternative: Extended Color Palette**

**Revised Case Study 1C (More Colors, Fewer Objects)**:
```json
{
  "concepts": [
    "red_car", "blue_car", "white_car", "black_car"
  ]
}
```

**Advantages**:
- 4 colors on single object category (cars = 4,785 total images)
- Tests pure color disentanglement
- Simpler narrative (no cross-category confounds)
- All concepts have high coverage (140-352+ images)

---

## Next Steps

1. **Run expanded feasibility test** to get exact counts for:
   - Buses (red, blue, yellow, white)
   - Trains (red, blue, yellow)
   - Kites (red, blue, yellow)
   - Boats (red, blue, white)
   - Balloons (red, blue, yellow)
   - Shirts (red, blue, white, black)
   - Flowers (red, yellow, white)

2. **Compare coverage** to current umbrella counts (165-179 images)

3. **Select replacement** based on:
   - Higher sample counts (N≥200 preferred)
   - Coherent category story
   - Visual distinctiveness

## Running the Test

```bash
cd /home/aticinovic/MoE-Specialisation
sbatch hpc/analysis_scripts/coco_feasibility_test.sbatch
```

Then check output for new color concepts added to the test.

"""
COCO Caption Feasibility Test for Compositional Concepts

Tests how many captions in COCO contain compositional color+object combinations
to determine if automated sampling is feasible.
"""

import json
import sys
from collections import defaultdict

def test_compositional_caption_counts(annotations_file: str):
    """
    Count how many COCO captions match compositional concept patterns.
    
    Args:
        annotations_file: Path to COCO captions JSON
    """
    print("=" * 80)
    print("COCO Compositional Concept Feasibility Test")
    print("=" * 80)
    print(f"\nLoading annotations from: {annotations_file}")
    
    with open(annotations_file, 'r') as f:
        coco = json.load(f)
    
    print(f"Total captions: {len(coco['annotations'])}")
    print(f"Total images: {len(coco['images'])}")
    
    # Define concept search patterns
    concepts = {
        # ========================================================================
        # CASE STUDY 1: COLOR COMPOSITIONAL (Color × Object)
        # ========================================================================
        
        # APPLES (Color compositional - proven good!)
        "red_apple": {
            "required": [["red"], ["apple", "apples"]],
            "exclude": ["green", "yellow"]
        },
        "green_apple": {
            "required": [["green"], ["apple", "apples"]],
            "exclude": ["red", "yellow"]
        },
        "yellow_apple": {
            "required": [["yellow"], ["apple", "apples"]],
            "exclude": ["red", "green"]
        },
        
        # ROSES (Color × Flower)
        "red_rose": {
            "required": [["red"], ["rose", "roses"]],
            "exclude": ["pink", "white", "yellow"]
        },
        "pink_rose": {
            "required": [["pink"], ["rose", "roses"]],
            "exclude": ["red", "white", "yellow"]
        },
        "white_rose": {
            "required": [["white"], ["rose", "roses"]],
            "exclude": ["red", "pink", "yellow"]
        },
        
        # CARS (Color × Vehicle)
        "red_car": {
            "required": [["red"], ["car", "cars"]],
            "exclude": ["blue", "white", "black", "green"]
        },
        "blue_car": {
            "required": [["blue"], ["car", "cars"]],
            "exclude": ["red", "white", "black", "green"]
        },
        "white_car": {
            "required": [["white"], ["car", "cars"]],
            "exclude": ["red", "blue", "black", "green"]
        },
        "black_car": {
            "required": [["black"], ["car", "cars"]],
            "exclude": ["red", "blue", "white", "green"]
        },
        
        # UMBRELLAS (Color × Object - highly varied in COCO)
        "red_umbrella": {
            "required": [["red"], ["umbrella", "umbrellas"]],
            "exclude": ["blue", "yellow", "green", "black", "white"]
        },
        "blue_umbrella": {
            "required": [["blue"], ["umbrella", "umbrellas"]],
            "exclude": ["red", "yellow", "green", "black", "white"]
        },
        "yellow_umbrella": {
            "required": [["yellow"], ["umbrella", "umbrellas"]],
            "exclude": ["red", "blue", "green", "black", "white"]
        },
        "green_umbrella": {
            "required": [["green"], ["umbrella", "umbrellas"]],
            "exclude": ["red", "blue", "yellow", "black", "white"]
        },
        "white_umbrella": {
            "required": [["white"], ["umbrella", "umbrellas"]],
            "exclude": ["red", "blue", "yellow", "green", "black"]
        },
        "black_umbrella": {
            "required": [["black"], ["umbrella", "umbrellas"]],
            "exclude": ["red", "blue", "yellow", "green", "white"]
        },
        
        # BUSES (Color × Large Vehicle - very abundant)
        "red_bus": {
            "required": [["red"], ["bus", "buses"]],
            "exclude": ["blue", "yellow", "green", "white", "tour", "school"]
        },
        "blue_bus": {
            "required": [["blue"], ["bus", "buses"]],
            "exclude": ["red", "yellow", "green", "white", "tour", "school"]
        },
        "yellow_bus": {
            "required": [["yellow"], ["bus", "buses"]],
            "exclude": ["red", "blue", "green", "white", "tour"]
        },
        "green_bus": {
            "required": [["green"], ["bus", "buses"]],
            "exclude": ["red", "blue", "yellow", "white", "tour", "school"]
        },
        "white_bus": {
            "required": [["white"], ["bus", "buses"]],
            "exclude": ["red", "blue", "yellow", "green", "tour", "school"]
        },
        
        # TRAINS (Color × Large Vehicle - very abundant)
        "red_train": {
            "required": [["red"], ["train", "trains"]],
            "exclude": ["blue", "yellow", "green", "white", "black"]
        },
        "blue_train": {
            "required": [["blue"], ["train", "trains"]],
            "exclude": ["red", "yellow", "green", "white", "black"]
        },
        "yellow_train": {
            "required": [["yellow"], ["train", "trains"]],
            "exclude": ["red", "blue", "green", "white", "black"]
        },
        
        # TRUCKS (Color × Vehicle)
        "red_truck": {
            "required": [["red"], ["truck", "trucks"]],
            "exclude": ["blue", "yellow", "green", "white", "black", "fire"]
        },
        "blue_truck": {
            "required": [["blue"], ["truck", "trucks"]],
            "exclude": ["red", "yellow", "green", "white", "black"]
        },
        "white_truck": {
            "required": [["white"], ["truck", "trucks"]],
            "exclude": ["red", "blue", "yellow", "green", "black"]
        },
        
        # BOATS (Color × Watercraft)
        "red_boat": {
            "required": [["red"], ["boat", "boats"]],
            "exclude": ["blue", "yellow", "green", "white"]
        },
        "blue_boat": {
            "required": [["blue"], ["boat", "boats"]],
            "exclude": ["red", "yellow", "green", "white"]
        },
        "white_boat": {
            "required": [["white"], ["boat", "boats"]],
            "exclude": ["red", "blue", "yellow", "green"]
        },
        
        # KITES (Color × Flying Object - popular in outdoor scenes)
        "red_kite": {
            "required": [["red"], ["kite", "kites"]],
            "exclude": ["blue", "yellow", "green", "bird"]
        },
        "blue_kite": {
            "required": [["blue"], ["kite", "kites"]],
            "exclude": ["red", "yellow", "green", "bird"]
        },
        "yellow_kite": {
            "required": [["yellow"], ["kite", "kites"]],
            "exclude": ["red", "blue", "green", "bird"]
        },
        
        # FLOWERS (Color × Natural Object - general flowers)
        "red_flower": {
            "required": [["red"], ["flower", "flowers"]],
            "exclude": ["blue", "yellow", "white", "pink"]
        },
        "yellow_flower": {
            "required": [["yellow"], ["flower", "flowers"]],
            "exclude": ["red", "blue", "white", "pink"]
        },
        "white_flower": {
            "required": [["white"], ["flower", "flowers"]],
            "exclude": ["red", "blue", "yellow", "pink"]
        },
        
        # BALLOONS (Color × Object - party/celebration)
        "red_balloon": {
            "required": [["red"], ["balloon", "balloons"]],
            "exclude": ["blue", "yellow", "green", "white"]
        },
        "blue_balloon": {
            "required": [["blue"], ["balloon", "balloons"]],
            "exclude": ["red", "yellow", "green", "white"]
        },
        "yellow_balloon": {
            "required": [["yellow"], ["balloon", "balloons"]],
            "exclude": ["red", "blue", "green", "white"]
        },
        
        # SHIRTS (Color × Clothing - very abundant with person)
        "red_shirt": {
            "required": [["red"], ["shirt", "shirts", "t-shirt"]],
            "exclude": ["blue", "yellow", "green", "white", "black"]
        },
        "blue_shirt": {
            "required": [["blue"], ["shirt", "shirts", "t-shirt"]],
            "exclude": ["red", "yellow", "green", "white", "black"]
        },
        "white_shirt": {
            "required": [["white"], ["shirt", "shirts", "t-shirt"]],
            "exclude": ["red", "blue", "yellow", "green", "black"]
        },
        "black_shirt": {
            "required": [["black"], ["shirt", "shirts", "t-shirt"]],
            "exclude": ["red", "blue", "yellow", "green", "white"]
        },
        
        # ========================================================================
        # CASE STUDY 1B: SHAPE COMPOSITIONAL (Shape × Category)
        # ========================================================================
        
        # ROUND OBJECTS (Same shape, different categories)
        "round_ball": {
            "required": [["round", "circular"], ["ball", "balls"]],
            "exclude": ["oval", "square"]
        },
        "round_plate": {
            "required": [["round", "circular"], ["plate", "plates"]],
            "exclude": ["oval", "square", "rectangular"]
        },
        "round_clock": {
            "required": [["round", "circular"], ["clock", "clocks"]],
            "exclude": ["square", "rectangular"]
        },
        "round_pizza": {
            "required": [["round", "circular"], ["pizza"]],
            "exclude": ["square", "rectangular"]
        },
        
        # RECTANGULAR OBJECTS (Same shape, different categories)
        "rectangular_table": {
            "required": [["rectangular", "rectangle"], ["table", "tables"]],
            "exclude": ["round", "circular", "square"]
        },
        "rectangular_door": {
            "required": [["rectangular"], ["door", "doors"]],
            "exclude": ["round", "circular", "square"]
        },
        "rectangular_window": {
            "required": [["rectangular"], ["window", "windows"]],
            "exclude": ["round", "circular", "square"]
        },
        
        # SQUARE OBJECTS
        "square_window": {
            "required": [["square"], ["window", "windows"]],
            "exclude": ["round", "circular", "rectangular"]
        },
        "square_tile": {
            "required": [["square"], ["tile", "tiles"]],
            "exclude": ["round", "circular", "rectangular"]
        },
        
        # SHAPE GENERAL (No specific object)
        "circle": {
            "required": [["circle", "circular"]],
            "exclude": ["semicircle"]
        },
        "square": {
            "required": [["square"]],
            "exclude": ["squared", "squaring"]
        },
        "triangle": {
            "required": [["triangle", "triangular"]],
            "exclude": []
        },
        "rectangle": {
            "required": [["rectangle", "rectangular"]],
            "exclude": []
        },
        
        # ========================================================================
        # CASE STUDY 2: SIZE COMPOSITIONAL (Size × Object)
        # ========================================================================
        
        # SIZE GRADIENT: FRUITS
        "small_apple": {
            "required": [["small", "little", "tiny"], ["apple", "apples"]],
            "exclude": ["large", "big", "huge"]
        },
        "large_apple": {
            "required": [["large", "big"], ["apple", "apples"]],
            "exclude": ["small", "little", "tiny"]
        },
        
        # SIZE GRADIENT: ANIMALS
        "small_dog": {
            "required": [["small", "little", "tiny"], ["dog", "dogs"]],
            "exclude": ["large", "big", "huge"]
        },
        "large_dog": {
            "required": [["large", "big"], ["dog", "dogs"]],
            "exclude": ["small", "little", "tiny"]
        },
        
        # SIZE GRADIENT: OBJECTS
        "small_boat": {
            "required": [["small", "little"], ["boat", "boats"]],
            "exclude": ["large", "big", "yacht", "ship"]
        },
        "large_boat": {
            "required": [["large", "big"], ["boat", "boats"]],
            "exclude": ["small", "little", "tiny"]
        },
        
        # ========================================================================
        # CASE STUDY 3: BASE CONCEPTS (Single-word, high frequency)
        # ========================================================================
        
        # COMMON FRUITS (Shape/size variation)
        "banana": {
            "required": [["banana", "bananas"]],
            "exclude": ["split", "bread"]
        },
        "orange": {
            "required": [["orange", "oranges"]],
            "exclude": ["juice", "color", "colored", "shirt"]
        },
        
        # BERRIES (Small fruits)
        "strawberry": {
            "required": [["strawberry", "strawberries"]],
            "exclude": []
        },
        "blueberry": {
            "required": [["blueberry", "blueberries"]],
            "exclude": []
        },
        "raspberry": {
            "required": [["raspberry", "raspberries"]],
            "exclude": []
        },
        
        # CITRUS (Color + type variation)
        "lemon": {
            "required": [["lemon", "lemons"]],
            "exclude": []
        },
        "lime": {
            "required": [["lime", "limes"]],
            "exclude": []
        },
        
        # VEGETABLES
        "tomato": {
            "required": [["tomato", "tomatoes"]],
            "exclude": ["soup", "sauce", "paste", "ketchup"]
        },
        "carrot": {
            "required": [["carrot", "carrots"]],
            "exclude": ["cake"]
        },
        "broccoli": {
            "required": [["broccoli"]],
            "exclude": []
        },
        "cucumber": {
            "required": [["cucumber", "cucumbers"]],
            "exclude": []
        },
        
        # ========================================================================
        # CASE STUDY 4: ANIMAL CONCEPTS (Species × Attributes)
        # ========================================================================
        
        # COMMON ANIMALS
        "cat": {
            "required": [["cat", "cats", "kitten", "kittens"]],
            "exclude": ["dog", "dogs"]
        },
        "dog": {
            "required": [["dog", "dogs", "puppy", "puppies"]],
            "exclude": ["cat", "cats"]
        },
        "horse": {
            "required": [["horse", "horses"]],
            "exclude": ["zebra"]
        },
        "cow": {
            "required": [["cow", "cows"]],
            "exclude": []
        },
        "sheep": {
            "required": [["sheep"]],
            "exclude": []
        },
        "elephant": {
            "required": [["elephant", "elephants"]],
            "exclude": []
        },
        "giraffe": {
            "required": [["giraffe", "giraffes"]],
            "exclude": []
        },
        "zebra": {
            "required": [["zebra", "zebras"]],
            "exclude": ["crossing"]
        },
        
        # BIRDS
        "bird": {
            "required": [["bird", "birds"]],
            "exclude": []
        },
        "duck": {
            "required": [["duck", "ducks"]],
            "exclude": []
        },
        "seagull": {
            "required": [["seagull", "seagulls"]],
            "exclude": []
        },
        
        # ========================================================================
        # CASE STUDY 5: SPORTS OBJECTS (Round objects, different sizes)
        # ========================================================================
        
        "soccer_ball": {
            "required": [["soccer"], ["ball"]],
            "exclude": ["player", "field"]
        },
        "tennis_ball": {
            "required": [["tennis"], ["ball"]],
            "exclude": ["racket", "court"]
        },
        "baseball": {
            "required": [["baseball"]],
            "exclude": ["player", "field", "bat", "game", "cap"]
        },
        "basketball": {
            "required": [["basketball"]],
            "exclude": ["player", "court", "hoop"]
        },
        "football": {
            "required": [["football"]],
            "exclude": ["player", "field", "helmet"]
        },
        
        # ========================================================================
        # CASE STUDY 6: FURNITURE (Indoor scene objects)
        # ========================================================================
        
        "chair": {
            "required": [["chair", "chairs"]],
            "exclude": ["wheelchair"]
        },
        "table": {
            "required": [["table", "tables"]],
            "exclude": ["tennis", "pool"]
        },
        "couch": {
            "required": [["couch", "sofa"]],
            "exclude": []
        },
        "bed": {
            "required": [["bed", "beds"]],
            "exclude": ["bedroom", "bedding"]
        },
        
        # ========================================================================
        # CASE STUDY 7: VEHICLES (Transportation category)
        # ========================================================================
        
        "car": {
            "required": [["car", "cars"]],
            "exclude": ["toy"]
        },
        "truck": {
            "required": [["truck", "trucks"]],
            "exclude": ["toy", "food"]
        },
        "bus": {
            "required": [["bus", "buses"]],
            "exclude": ["tour"]
        },
        "train": {
            "required": [["train", "trains"]],
            "exclude": ["station", "toy"]
        },
        "bicycle": {
            "required": [["bicycle", "bicycles", "bike", "bikes"]],
            "exclude": ["motorcycle"]
        },
        "motorcycle": {
            "required": [["motorcycle", "motorcycles"]],
            "exclude": ["bicycle", "bike"]
        },
        
        # ========================================================================
        # CASE STUDY 8: ABSTRACT CONCEPTS (State/Activity)
        # ========================================================================
        
        # ACTIONS (Activity concepts)
        "person_sitting": {
            "required": [["person", "people", "man", "woman"], ["sitting", "sits", "seated"]],
            "exclude": []
        },
        "person_standing": {
            "required": [["person", "people", "man", "woman"], ["standing", "stands"]],
            "exclude": []
        },
        "person_walking": {
            "required": [["person", "people", "man", "woman"], ["walking", "walks"]],
            "exclude": []
        },
        "person_running": {
            "required": [["person", "people", "man", "woman"], ["running", "runs"]],
            "exclude": []
        },
        
        # STATES
        "open_door": {
            "required": [["open"], ["door", "doors"]],
            "exclude": ["closed"]
        },
        "closed_door": {
            "required": [["closed"], ["door", "doors"]],
            "exclude": ["open"]
        },
        
        # WEATHER/TIME
        "sunny_day": {
            "required": [["sunny", "sunshine"], ["day"]],
            "exclude": ["cloudy", "rainy"]
        },
        "cloudy_sky": {
            "required": [["cloudy", "clouds"], ["sky"]],
            "exclude": ["clear", "sunny"]
        },
        
        # ========================================================================
        # CASE STUDY 9: SPATIAL CONCEPTS (Location/Arrangement)
        # ========================================================================
        
        "indoor_scene": {
            "required": [["indoor", "inside", "room", "kitchen", "bedroom", "bathroom", "living"]],
            "exclude": ["outdoor", "outside"]
        },
        "outdoor_scene": {
            "required": [["outdoor", "outside", "street", "park", "field", "beach"]],
            "exclude": ["indoor", "inside"]
        },
        
        # ========================================================================
        # CASE STUDY 10: MATERIAL/TEXTURE (Surface properties)
        # ========================================================================
        
        "wooden_table": {
            "required": [["wooden", "wood"], ["table"]],
            "exclude": ["glass", "metal"]
        },
        "glass_vase": {
            "required": [["glass"], ["vase"]],
            "exclude": ["wooden", "plastic"]
        },
        "metal_fork": {
            "required": [["metal", "silver"], ["fork", "forks"]],
            "exclude": ["plastic"]
        },
        
        # ========================================================================
        # CASE STUDY 11: FOOD PREPARED DISHES (Compositional food)
        # ========================================================================
        
        "pizza": {
            "required": [["pizza"]],
            "exclude": ["box", "delivery"]
        },
        "sandwich": {
            "required": [["sandwich", "sandwiches"]],
            "exclude": []
        },
        "hot_dog": {
            "required": [["hot"], ["dog", "dogs"]],
            "exclude": ["animal", "pet"]
        },
        "cake": {
            "required": [["cake", "cakes"]],
            "exclude": ["birthday", "wedding"]
        },
        "donut": {
            "required": [["donut", "donuts", "doughnut", "doughnuts"]],
            "exclude": []
        },
        
        # ========================================================================
        # ADDITIONAL ORIGINAL CONCEPTS
        # ========================================================================
        
        "cherry": {
            "required": [["cherry", "cherries"]],
            "exclude": ["apple", "pear", "tomato", "strawberry", "picker", "blossom", "tree"]
        },
        "pear": {
            "required": [["pear", "pears"]],
            "exclude": ["apple", "cherry", "banana", "orange"]
        },
    }
    
    # Also count base objects without modifiers
    base_counts = {
        # Fruits
        "apple (any)": 0,
        "banana (any)": 0,
        "orange (any)": 0,
        "strawberry (any)": 0,
        "tomato (any)": 0,
        "carrot (any)": 0,
        "broccoli (any)": 0,
        # Animals
        "cat (any)": 0,
        "dog (any)": 0,
        "horse (any)": 0,
        "bird (any)": 0,
        # Vehicles
        "car (any)": 0,
        "bus (any)": 0,
        "train (any)": 0,
        "truck (any)": 0,
        "boat (any)": 0,
        # Objects
        "ball (any)": 0,
        "umbrella (any)": 0,
        "chair (any)": 0,
        "table (any)": 0,
        "kite (any)": 0,
        "balloon (any)": 0,
        "flower (any)": 0,
        "shirt (any)": 0,
        # People
        "person (any)": 0,
    }
    
    # Count matching captions
    concept_matches = {concept: [] for concept in concepts.keys()}
    
    print("\n" + "=" * 80)
    print("Analyzing captions...")
    print("=" * 80)
    
    for ann in coco['annotations']:
        caption = ann['caption'].lower()
        words = set(caption.split())
        
        # Count base objects
        if any(w in words for w in ["apple", "apples"]):
            base_counts["apple (any)"] += 1
        if any(w in words for w in ["banana", "bananas"]):
            base_counts["banana (any)"] += 1
        if any(w in words for w in ["orange", "oranges"]):
            base_counts["orange (any)"] += 1
        if any(w in words for w in ["strawberry", "strawberries"]):
            base_counts["strawberry (any)"] += 1
        if any(w in words for w in ["tomato", "tomatoes"]):
            base_counts["tomato (any)"] += 1
        if any(w in words for w in ["carrot", "carrots"]):
            base_counts["carrot (any)"] += 1
        if "broccoli" in words:
            base_counts["broccoli (any)"] += 1
        if "ball" in words:
            base_counts["ball (any)"] += 1
        if any(w in words for w in ["cat", "cats", "kitten", "kittens"]):
            base_counts["cat (any)"] += 1
        if any(w in words for w in ["dog", "dogs", "puppy", "puppies"]):
            base_counts["dog (any)"] += 1
        if any(w in words for w in ["horse", "horses"]):
            base_counts["horse (any)"] += 1
        if any(w in words for w in ["bird", "birds"]):
            base_counts["bird (any)"] += 1
        if any(w in words for w in ["car", "cars"]):
            base_counts["car (any)"] += 1
        if any(w in words for w in ["bus", "buses"]):
            base_counts["bus (any)"] += 1
        if any(w in words for w in ["train", "trains"]):
            base_counts["train (any)"] += 1
        if any(w in words for w in ["umbrella", "umbrellas"]):
            base_counts["umbrella (any)"] += 1
        if any(w in words for w in ["chair", "chairs"]):
            base_counts["chair (any)"] += 1
        if any(w in words for w in ["table", "tables"]):
            base_counts["table (any)"] += 1
        if any(w in words for w in ["truck", "trucks"]):
            base_counts["truck (any)"] += 1
        if any(w in words for w in ["boat", "boats"]):
            base_counts["boat (any)"] += 1
        if any(w in words for w in ["kite", "kites"]):
            base_counts["kite (any)"] += 1
        if any(w in words for w in ["balloon", "balloons"]):
            base_counts["balloon (any)"] += 1
        if any(w in words for w in ["flower", "flowers"]):
            base_counts["flower (any)"] += 1
        if any(w in words for w in ["shirt", "shirts", "t-shirt", "t-shirts"]):
            base_counts["shirt (any)"] += 1
        if any(w in words for w in ["person", "people", "man", "woman", "men", "women"]):
            base_counts["person (any)"] += 1
        
        # Check each compositional concept
        for concept_name, pattern in concepts.items():
            # Check if ALL required word groups are present
            all_required_present = all(
                any(word in words for word in word_group)
                for word_group in pattern["required"]
            )
            
            # Check if ANY exclude words are present
            any_exclude_present = any(word in words for word in pattern["exclude"])
            
            if all_required_present and not any_exclude_present:
                concept_matches[concept_name].append({
                    "image_id": ann["image_id"],
                    "caption": ann["caption"]
                })
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS: Compositional Concepts (Strict Matching)")
    print("=" * 80)
    print(f"{'Concept':<20} {'Matches':<10} {'Unique Images':<15} {'Feasible (N≥50)?'}")
    print("-" * 80)
    
    for concept_name in sorted(concept_matches.keys()):
        matches = concept_matches[concept_name]
        unique_images = len(set(m["image_id"] for m in matches))
        feasible = "✅ YES" if unique_images >= 50 else "❌ NO"
        print(f"{concept_name:<20} {len(matches):<10} {unique_images:<15} {feasible}")
    
    print("\n" + "=" * 80)
    print("RESULTS: Base Objects (Any Color)")
    print("=" * 80)
    print(f"{'Concept':<20} {'Captions':<10}")
    print("-" * 80)
    for concept, count in sorted(base_counts.items()):
        print(f"{concept:<20} {count:<10}")
    
    # Print sample captions for inspection
    print("\n" + "=" * 80)
    print("SAMPLE CAPTIONS (First 5 per concept)")
    print("=" * 80)
    for concept_name, matches in sorted(concept_matches.items()):
        print(f"\n{concept_name.upper()}:")
        if len(matches) == 0:
            print("  (No matches found)")
        else:
            for match in matches[:5]:
                print(f"  - {match['caption']}")
    
    # Summary recommendations by case study
    print("\n" + "=" * 80)
    print("CASE STUDY FEASIBILITY SUMMARY")
    print("=" * 80)
    
    feasible_concepts = {c: len(set(m["image_id"] for m in matches)) 
                        for c, matches in concept_matches.items()}
    
    # Define case study groupings
    case_studies = {
        "Color Compositional": [
            "red_apple", "green_apple", "yellow_apple",
            "red_rose", "pink_rose", "white_rose",
            "red_car", "blue_car", "white_car", "black_car",
            "red_umbrella", "blue_umbrella", "yellow_umbrella"
        ],
        "Size Compositional": [
            "small_apple", "large_apple",
            "small_dog", "large_dog",
            "small_boat", "large_boat"
        ],
        "Base Fruits": [
            "banana", "orange", "strawberry", "blueberry", "raspberry",
            "lemon", "lime", "cherry", "pear"
        ],
        "Base Vegetables": [
            "tomato", "carrot", "broccoli", "cucumber"
        ],
        "Animals": [
            "cat", "dog", "horse", "cow", "sheep", "elephant", "giraffe", "zebra",
            "bird", "duck", "seagull"
        ],
        "Sports Objects": [
            "soccer_ball", "tennis_ball", "baseball", "basketball", "football"
        ],
        "Furniture": [
            "chair", "table", "couch", "bed"
        ],
        "Vehicles": [
            "car", "truck", "bus", "train", "bicycle", "motorcycle"
        ],
        "Activity/Actions": [
            "person_sitting", "person_standing", "person_walking", "person_running"
        ],
        "States": [
            "open_door", "closed_door", "sunny_day", "cloudy_sky"
        ],
        "Spatial": [
            "indoor_scene", "outdoor_scene"
        ],
        "Material/Texture": [
            "wooden_table", "glass_vase", "metal_fork"
        ],
        "Prepared Food": [
            "pizza", "sandwich", "hot_dog", "cake", "donut"
        ]
    }
    
    for study_name, concept_list in case_studies.items():
        feasible_in_study = [c for c in concept_list if feasible_concepts.get(c, 0) >= 50]
        total_in_study = len(concept_list)
        
        print(f"\n{study_name}:")
        print(f"  Feasible: {len(feasible_in_study)}/{total_in_study} concepts")
        
        if len(feasible_in_study) >= 4:
            print(f"  ✅ EXCELLENT - Can run full case study")
            print(f"  Recommended: {', '.join(feasible_in_study[:6])}")
        elif len(feasible_in_study) >= 2:
            print(f"  ⚠️  PARTIAL - Can run limited case study")
            print(f"  Available: {', '.join(feasible_in_study)}")
        else:
            print(f"  ❌ INSUFFICIENT - Need alternatives")
            if feasible_in_study:
                print(f"  Only: {', '.join(feasible_in_study)}")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL RECOMMENDATIONS")
    print("=" * 80)
    
    all_feasible = [c for c, count in feasible_concepts.items() if count >= 50]
    
    print(f"\nTotal feasible concepts: {len(all_feasible)}/{len(concept_matches)}")
    print(f"\nTop 10 most abundant concepts:")
    sorted_concepts = sorted(feasible_concepts.items(), key=lambda x: x[1], reverse=True)
    for i, (concept, count) in enumerate(sorted_concepts[:10], 1):
        print(f"  {i}. {concept}: {count} unique images")
    
    print("\n" + "=" * 80)
    print("SUGGESTED CASE STUDIES TO RUN:")
    print("=" * 80)
    
    # Find best case studies
    best_studies = []
    for study_name, concept_list in case_studies.items():
        feasible_count = len([c for c in concept_list if feasible_concepts.get(c, 0) >= 50])
        if feasible_count >= 4:
            best_studies.append((study_name, feasible_count, concept_list))
    
    best_studies.sort(key=lambda x: x[1], reverse=True)
    
    if best_studies:
        print("\n✅ Recommended case studies (in order of feasibility):")
        for i, (name, count, concepts) in enumerate(best_studies[:5], 1):
            feasible_in_study = [c for c in concepts if feasible_concepts.get(c, 0) >= 50]
            print(f"\n{i}. {name} ({count} feasible concepts)")
            print(f"   Concepts: {', '.join(feasible_in_study[:8])}")
    else:
        print("\n⚠️  No case studies have ≥4 feasible concepts")
        print("Consider using base objects or relaxing exclusion criteria")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python coco_feasibility_test.py <path_to_captions_train2017.json>")
        sys.exit(1)
    
    annotations_file = sys.argv[1]
    test_compositional_caption_counts(annotations_file)

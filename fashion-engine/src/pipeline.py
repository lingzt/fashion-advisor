"""
Fashion Recommender Pipeline
=========================
Complete pipeline: User Input → LLM Parse → RAG Retrieval → Rule Assembly → LLM Explain

Architecture:
1. User Input (自然语言需求)
2. LLM Parse → JSON Schema
3. RAG Retrieval → Candidate Items
4. Rule Assembly → Complete Outfit
5. LLM Explain → Final Response
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import os

# LLM Factory (Free Models)
try:
    from src.llm_factory import LLMFactory
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class Intent(Enum):
    """Request intent types"""
    OUTFIT_SET = "outfit_set"
    SEARCH = "search"
    ADVICE = "style_advice"
    COMPARE = "compare"


@dataclass
class RequestSchema:
    """Structured request from LLM parsing"""
    intent: Intent
    occasion: Optional[str] = None
    season: Optional[str] = None
    style_preference: Optional[str] = None
    color_preference: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    avoid_colors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "occasion": self.occasion,
            "season": self.season,
            "style_preference": self.style_preference,
            "color_preference": self.color_preference,
            "constraints": self.constraints,
            "budget_min": self.budget_min,
            "budget_max": self.budget_max,
            "avoid_colors": self.avoid_colors,
        }


@dataclass
class OutfitItem:
    """Fashion item representation"""
    item_id: str
    name: str
    category: str  # TOP, BOTTOM, DRESS, FOOTWEAR, ACCESSORY
    subcategory: str
    color: str
    color_hex: str
    formality: int  # 1-5
    warmth: int  # 1-5
    season: str
    occasions: List[str]
    styles: List[str]
    price: Optional[int] = None
    image_url: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "name": self.name,
            "category": self.category,
            "color": self.color,
            "color_hex": self.color_hex,
            "formality": self.formality,
        }


@dataclass
class ResponseSchema:
    """Final response schema"""
    outfits: List[Dict] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    image_prompts: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class LLMParse:
    """Step 2: LLM parses natural language to RequestSchema"""
    
    PARSE_PROMPT = """
You are a fashion stylist assistant. Parse the user's request into structured JSON.

User Request: {user_input}

Output JSON format:
{{
    "intent": "outfit_set" | "search" | "advice" | "compare",
    "occasion": "casual" | "work" | "date" | "party" | "formal" | "sport" | null,
    "season": "spring" | "summer" | "fall" | "winter" | "all" | null,
    "style_preference": "casual" | "formal" | "trendy" | "athletic" | "minimalist" | "vintage" | null,
    "color_preference": ["white", "black", "navy", "beige", "blue", ...],
    "constraints": ["no_leather", "vegan", "sustainable", ...],
    "budget_min": 1 | 2 | 3 | 4 | 5 | null,  # 1=$0-50, 2=$50-100, 3=$100-200, 4=$200-500, 5=$500+
    "budget_max": 1 | 2 | 3 | 4 | 5 | null,
    "avoid_colors": ["red", "orange", ...]
}}

Rules:
- Extract ALL mentioned colors
- Identify occasion from context (date night, work meeting, etc.)
- Infer season from context if not explicit
- Note any constraints (no leather, vegan, budget limits)
- Default budget to null if not mentioned

Output ONLY valid JSON, no additional text.
"""
    
    def __init__(self):
        self.llm = LLMFactory().get_default() if LLM_AVAILABLE else None
    
    def parse(self, user_input: str) -> RequestSchema:
        """Parse user input to structured request"""
        if self.llm:
            prompt = self.PARSE_PROMPT.format(user_input=user_input)
            response = self.llm.generate(prompt, max_tokens=500)
            
            try:
                # Extract JSON from response
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                
                data = json.loads(json_str)
                
                return RequestSchema(
                    intent=Intent(data.get("intent", "outfit_set")),
                    occasion=data.get("occasion"),
                    season=data.get("season"),
                    style_preference=data.get("style_preference"),
                    color_preference=data.get("color_preference", []),
                    constraints=data.get("constraints", []),
                    budget_min=data.get("budget_min"),
                    budget_max=data.get("budget_max"),
                    avoid_colors=data.get("avoid_colors", []),
                )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Parse error: {e}, using defaults")
        
        # Fallback: return default request
        return RequestSchema(intent=Intent.OUTFIT_SET)


class ItemDatabase:
    """Fashion item database with metadata"""
    
    def __init__(self):
        self.items: List[OutfitItem] = self._init_db()
    
    def _init_db(self) -> List[OutfitItem]:
        return [
            # TOPS
            OutfitItem("T1", "Classic White Tee", "TOP", "T-Shirt", 
                      "WHITE", "#FFFFFF", 1, 1, "all",
                      ["casual", "date"], ["casual", "minimalist"], 25),
            OutfitItem("T2", "Navy Oxford Shirt", "TOP", "Shirt",
                      "NAVY", "#000080", 4, 2, "all",
                      ["work", "date", "formal"], ["formal", "minimalist"], 85),
            OutfitItem("T3", "Beige Cable Sweater", "TOP", "Sweater",
                      "BEIGE", "#F5F5DC", 2, 4, "fall",
                      ["casual", "date"], ["casual", "vintage"], 120),
            OutfitItem("T4", "Black Leather Jacket", "TOP", "Jacket",
                      "BLACK", "#000000", 3, 3, "all",
                      ["casual", "date", "party"], ["trendy", "casual"], 250),
            OutfitItem("T5", "White Button-Down", "TOP", "Shirt",
                      "WHITE", "#FFFFFF", 3, 1, "all",
                      ["work", "date"], ["formal", "minimalist"], 95),
            OutfitItem("T6", "Olive Hoodie", "TOP", "Hoodie",
                      "OLIVE", "#808000", 1, 3, "fall",
                      ["casual", "sport"], ["athletic", "casual"], 65),
            OutfitItem("T7", "Red Cashmere Sweater", "TOP", "Sweater",
                      "RED", "#FF0000", 2, 4, "winter",
                      ["casual", "date"], ["elegant"], 180),
            OutfitItem("T8", "Black Turtleneck", "TOP", "Shirt",
                      "BLACK", "#000000", 4, 3, "winter",
                      ["work", "formal", "date"], ["formal", "minimalist"], 110),
            
            # BOTTOMS
            OutfitItem("B1", "Blue Straight Jeans", "BOTTOM", "Jeans",
                      "BLUE", "#0000FF", 2, 2, "all",
                      ["casual", "date"], ["casual"], 75),
            OutfitItem("B2", "Black Dress Pants", "BOTTOM", "Trousers",
                      "BLACK", "#000000", 5, 1, "all",
                      ["work", "formal", "date"], ["formal"], 120),
            OutfitItem("B3", "Beige Chinos", "BOTTOM", "Chinos",
                      "BEIGE", "#F5F5DC", 3, 2, "spring",
                      ["casual", "work", "date"], ["casual", "preppy"], 95),
            OutfitItem("B4", "White Linen Pants", "BOTTOM", "Pants",
                      "WHITE", "#FFFFFF", 2, 1, "summer",
                      ["casual", "date"], ["casual"], 85),
            OutfitItem("B5", "Gray Sweatpants", "BOTTOM", "Sweatpants",
                      "GRAY", "#808080", 1, 3, "winter",
                      ["casual", "sport"], ["athletic", "casual"], 45),
            
            # DRESSES
            OutfitItem("D1", "Black Midi Dress", "DRESS", "Midi Dress",
                      "BLACK", "#000000", 4, 2, "all",
                      ["date", "party", "formal"], ["elegant", "minimalist"], 180),
            OutfitItem("D2", "Floral Summer Dress", "DRESS", "Summer Dress",
                      "WHITE", "#FFFFFF", 2, 1, "summer",
                      ["casual", "date"], ["casual", "feminine"], 95),
            OutfitItem("D3", "Navy Wrap Dress", "DRESS", "Wrap Dress",
                      "NAVY", "#000080", 3, 2, "all",
                      ["work", "date"], ["elegant", "professional"], 150),
            
            # FOOTWEAR
            OutfitItem("F1", "White Sneakers", "FOOTWEAR", "Sneakers",
                      "WHITE", "#FFFFFF", 1, 1, "all",
                      ["casual", "sport"], ["casual", "athletic"], 85),
            OutfitItem("F2", "Brown Loafers", "FOOTWEAR", "Loafers",
                      "BROWN", "#8B4513", 4, 2, "all",
                      ["work", "formal", "date"], ["formal", "minimalist"], 145),
            OutfitItem("F3", "Black Chelsea Boots", "FOOTWEAR", "Boots",
                      "BLACK", "#000000", 3, 3, "fall",
                      ["casual", "date"], ["trendy", "casual"], 195),
            OutfitItem("F4", "White Slip-Ons", "FOOTWEAR", "Slip-Ons",
                      "WHITE", "#FFFFFF", 1, 1, "summer",
                      ["casual"], ["casual"], 55),
            OutfitItem("F5", "Black Heels", "FOOTWEAR", "Heels",
                      "BLACK", "#000000", 4, 1, "all",
                      ["work", "date", "party"], ["elegant"], 125),
            
            # ACCESSORIES
            OutfitItem("A1", "Silver Necklace", "ACCESSORY", "Necklace",
                      "GRAY", "#808080", 2, 1, "all",
                      ["date", "party", "casual"], ["trendy", "minimalist"], 45),
            OutfitItem("A2", "Leather Belt", "ACCESSORY", "Belt",
                      "BROWN", "#8B4513", 3, 1, "all",
                      ["work", "formal", "casual"], ["formal", "casual"], 55),
            OutfitItem("A3", "Black Watch", "ACCESSORY", "Watch",
                      "BLACK", "#000000", 3, 1, "all",
                      ["work", "date"], ["formal", "minimalist"], 150),
            OutfitItem("A4", "Beige Tote Bag", "ACCESSORY", "Bag",
                      "BEIGE", "#F5F5DC", 2, 1, "all",
                      ["casual", "work"], ["casual", "minimalist"], 85),
        ]
    
    def search(self, query: str = "", filters: Dict = None) -> List[OutfitItem]:
        """Search items with filters"""
        results = self.items
        
        if filters:
            if filters.get("category"):
                results = [i for i in results if i.category == filters["category"]]
            if filters.get("color"):
                results = [i for i in results if i.color in filters["color"]]
            if filters.get("season"):
                results = [i for i in results if filters["season"] in [i.season, "all"]]
            if filters.get("occasion"):
                results = [i for i in results if filters["occasion"] in i.occasions]
            if filters.get("style"):
                results = [i for i in results if filters["style"] in i.styles]
            if filters.get("max_price"):
                results = [i for i in results if not i.price or i.price <= filters["max_price"]]
            if filters.get("avoid_colors"):
                results = [i for i in results if i.color not in filters["avoid_colors"]]
            if filters.get("constraints"):
                if "no_leather" in filters["constraints"]:
                    results = [i for i in results if "leather" not in i.name.lower()]
                if "vegan" in filters["constraints"]:
                    # Simple vegan check (no leather, wool, silk)
                    results = [i for i in results if "leather" not in i.name.lower() 
                              and "wool" not in i.name.lower()
                              and "silk" not in i.name.lower()
                              and "cashmere" not in i.name.lower()]
        
        return results


class RuleAssembler:
    """Step 4: Deterministic outfit assembly rules"""
    
    def __init__(self):
        self.db = ItemDatabase()
    
    def assemble(self, request: RequestSchema, candidates: List[OutfitItem]) -> List[Dict]:
        """Assemble complete outfits from candidates"""
        outfits = []
        
        # Separate by category
        tops = [i for i in candidates if i.category == "TOP"]
        bottoms = [i for i in candidates if i.category == "BOTTOM"]
        dresses = [i for i in candidates if i.category == "DRESS"]
        footwear = [i for i in candidates if i.category == "FOOTWEAR"]
        accessories = [i for i in candidates if i.category == "ACCESSORY"]
        
        # Generate TOP + BOTTOM combinations
        for top in tops[:5]:  # Limit candidates
            for bottom in bottoms[:5]:
                if self._is_compatible(top, bottom, request):
                    outfit = self._create_outfit(request, top, bottom, footwear, accessories)
                    if outfit:
                        outfits.append(outfit)
        
        # Generate DRESS only outfits
        for dress in dresses[:5]:
            outfit = self._create_dress_outfit(request, dress, footwear, accessories)
            if outfit:
                outfits.append(outfit)
        
        # Sort by compatibility score
        outfits.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return outfits[:5]  # Return top 5 outfits
    
    def _is_compatible(self, top: OutfitItem, bottom: OutfitItem, request: RequestSchema) -> bool:
        """Check if two items are compatible"""
        # Color harmony check
        if request.avoid_colors:
            if top.color in request.avoid_colors or bottom.color in request.avoid_colors:
                return False
        
        # Formality should be similar
        formality_diff = abs(top.formality - bottom.formality)
        if formality_diff > 2:
            return False
        
        # Season compatibility
        if request.season and request.season != "all":
            if top.season not in [request.season, "all"]:
                return False
            if bottom.season not in [request.season, "all"]:
                return False
        
        return True
    
    def _create_outfit(self, request: RequestSchema, top: OutfitItem, bottom: OutfitItem,
                       footwear: List[OutfitItem], accessories: List[OutfitItem]) -> Optional[Dict]:
        """Create complete outfit from top + bottom"""
        # Select matching footwear
        shoes = None
        for shoe in footwear:
            if abs(shoe.formality - top.formality) <= 1:
                shoes = shoe
                break
        
        # Select matching accessory
        acc = None
        for a in accessories:
            if abs(a.formality - top.formality) <= 1:
                acc = a
                break
        
        # Calculate score
        formality_score = 1.0 - abs(top.formality - bottom.formality) / 5
        color_score = 1.0 if top.color == bottom.color else 0.8
        occasion_score = 1.0 if request.occasion in top.occasions else 0.5
        total_score = round(formality_score * color_score * occasion_score * 100, 1)
        
        return {
            "type": "top_bottom",
            "score": total_score,
            "items": {
                "top": top.to_dict(),
                "bottom": bottom.to_dict(),
                "footwear": shoes.to_dict() if shoes else None,
                "accessory": acc.to_dict() if acc else None,
            },
            "colors": [top.color, bottom.color],
            "total_price": sum([
                top.price or 0,
                bottom.price or 0,
                shoes.price if shoes else 0,
                acc.price if acc else 0,
            ]),
        }
    
    def _create_dress_outfit(self, request: RequestSchema, dress: OutfitItem,
                           footwear: List[OutfitItem], accessories: List[OutfitItem]) -> Optional[Dict]:
        """Create outfit from dress only"""
        # Select matching footwear
        shoes = None
        for shoe in footwear:
            if abs(shoe.formality - dress.formality) <= 1:
                shoes = shoe
                break
        
        # Select matching accessory
        acc = None
        for a in accessories:
            if abs(a.formality - dress.formality) <= 1:
                acc = a
                break
        
        occasion_score = 1.0 if request.occasion in dress.occasions else 0.5
        total_score = round(occasion_score * 100, 1)
        
        return {
            "type": "dress",
            "score": total_score,
            "items": {
                "dress": dress.to_dict(),
                "footwear": shoes.to_dict() if shoes else None,
                "accessory": acc.to_dict() if acc else None,
            },
            "colors": [dress.color],
            "total_price": sum([
                dress.price or 0,
                shoes.price if shoes else 0,
                acc.price if acc else 0,
            ]),
        }


class LLMExplainer:
    """Step 5: LLM generates explanations"""
    
    EXPLAIN_PROMPT = """
You are a fashion stylist. Explain why this outfit works.

Outfit Details:
{outfit_json}

User Request:
- Occasion: {occasion}
- Style: {style}
- Preferences: {preferences}

Provide:
1. Why these pieces work together (color harmony, formality, style)
2. When to wear this outfit
3. One styling tip

Keep response concise: 2-3 sentences.
Output JSON format:
{{
    "reason": "Brief explanation",
    "when_to_wear": "When this outfit works best",
    "styling_tip": "One practical tip"
}}
"""
    
    def __init__(self):
        self.llm = LLMFactory().get_default() if LLM_AVAILABLE else None
    
    def explain(self, outfit: Dict, request: RequestSchema) -> Dict:
        """Generate explanation for outfit"""
        if self.llm:
            prompt = self.EXPLAIN_PROMPT.format(
                outfit_json=json.dumps(outfit, indent=2),
                occasion=request.occasion or "general",
                style=request.style_preference or "any",
                preferences=", ".join(request.color_preference) if request.color_preference else "none"
            )
            
            response = self.llm.generate(prompt, max_tokens=300)
            
            try:
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                
                return json.loads(json_str)
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Fallback explanation
        colors = outfit.get("colors", [])
        return {
            "reason": f"A cohesive {', '.join(colors)} color scheme creates visual harmony.",
            "when_to_wear": f"Perfect for {request.occasion or 'any occasion'}.",
            "styling_tip": "Add a statement piece to elevate the look."
        }


class FashionPipeline:
    """Complete Fashion Recommender Pipeline"""
    
    def __init__(self):
        self.llm_parser = LLMParse()
        self.db = ItemDatabase()
        self.assembler = RuleAssembler()
        self.explainer = LLMExplainer()
    
    def run(self, user_input: str) -> ResponseSchema:
        """
        Execute complete pipeline:
        1. Parse user input → RequestSchema
        2. Generate retrieval query
        3. Retrieve candidates
        4. Assemble outfits
        5. Generate explanations
        """
        print(f"\n📝 Step 1: User Input")
        print(f"   \"{user_input}\"")
        
        # Step 1: Parse
        print(f"\n🔍 Step 2: LLM Parsing...")
        request = self.llm_parser.parse(user_input)
        print(f"   Intent: {request.intent.value}")
        print(f"   Occasion: {request.occasion}")
        print(f"   Season: {request.season}")
        print(f"   Style: {request.style_preference}")
        print(f"   Colors: {request.color_preference}")
        print(f"   Constraints: {request.constraints}")
        
        # Step 2: Build search filters
        print(f"\n📦 Step 3: RAG Retrieval...")
        filters = {
            "occasion": request.occasion,
            "season": request.season,
            "style": request.style_preference,
            "color": request.color_preference if request.color_preference else None,
            "avoid_colors": request.avoid_colors,
            "constraints": request.constraints,
        }
        filters = {k: v for k, v in filters.items() if v}
        
        candidates = self.db.search(filters=filters)
        print(f"   Retrieved {len(candidates)} candidates")
        
        # Step 3: Assemble
        print(f"\n🎨 Step 4: Rule Assembly...")
        outfits = self.assembler.assemble(request, candidates)
        print(f"   Generated {len(outfits)} complete outfits")
        
        # Step 4: Explain
        print(f"\n💬 Step 5: LLM Explanation...")
        reasons = []
        for outfit in outfits[:3]:  # Explain top 3
            explanation = self.explainer.explain(outfit, request)
            reasons.append(explanation)
        
        # Generate image prompts
        image_prompts = self._generate_image_prompts(outfits)
        
        return ResponseSchema(
            outfits=outfits,
            reasons=[r.get("reason", "") for r in reasons],
            image_prompts=image_prompts,
            suggestions=[r.get("styling_tip", "") for r in reasons],
        )
    
    def _generate_image_prompts(self, outfits: List[Dict]) -> List[str]:
        """Generate image generation prompts"""
        prompts = []
        for outfit in outfits[:3]:
            items = outfit.get("items", {})
            names = []
            colors = []
            
            for key, item in items.items():
                if item:
                    names.append(item.get("name", ""))
                    colors.append(item.get("color", ""))
            
            prompt = (
                f"Fashion flat lay photography, {', '.join(names)}, "
                f"colors: {', '.join(colors)}, "
                f"minimalist studio background, professional e-commerce photography"
            )
            prompts.append(prompt)
        
        return prompts


def demo_pipeline():
    """Demo the complete pipeline"""
    
    print("\n" + "🎯"*60)
    print("   FASHION RECOMMENDER PIPELINE DEMO")
    print("   Complete: Parse → Retrieve → Assemble → Explain")
    print("🎯"*60)
    
    pipeline = FashionPipeline()
    
    # Demo requests
    test_inputs = [
        "I need an outfit for a date night, prefer minimalist style, no leather",
        "What should I wear to work? I like neutral colors, budget around $100",
        "Spring wedding guest, elegant style, avoid black",
    ]
    
    for user_input in test_inputs:
        print("\n" + "="*60)
        response = pipeline.run(user_input)
        
        print(f"\n✨ Generated Outfits: {len(response.outfits)}")
        for i, outfit in enumerate(response.outfits[:2], 1):
            items = outfit.get("items", {})
            print(f"\n  Outfit {i} (Score: {outfit.get('score', 0)})")
            for key, item in items.items():
                if item:
                    print(f"    - {item['name']} ({item['color']})")
        
        if response.reasons:
            print(f"\n💡 Style Notes:")
            for reason in response.reasons[:2]:
                print(f"   • {reason}")
    
    print("\n" + "🎯"*60)
    print("   Pipeline Demo Complete!")
    print("🎯"*60)


if __name__ == "__main__":
    demo_pipeline()

"""
Deterministic Outfit Assembler Rule Engine
==========================================
Lightweight rule layer for outfit generation.
Uses predefined rules for color matching, style compatibility,
and seasonal appropriateness.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json


class Color(Enum):
    """Color palette with hex values"""
    # Neutral
    BLACK = "#000000"
    WHITE = "#FFFFFF"
    GRAY = "#808080"
    BEIGE = "#F5F5DC"
    CAMEL = "#C19A6B"
    NAVY = "#000080"
    BROWN = "#8B4513"
    
    # Warm
    RED = "#FF0000"
    ORANGE = "#FFA500"
    CORAL = "#FF7F50"
    PEACH = "#FFE5B4"
    YELLOW = "#FFFF00"
    GOLD = "#FFD700"
    
    # Cool
    BLUE = "#0000FF"
    TEAL = "#008080"
    TURQUOISE = "#40E0D0"
    COBALT = "#0047AB"
    INDIGO = "#4B0082"
    
    # Pastel
    PINK = "#FFB6C1"
    LAVENDER = "#E6E6FA"
    MINT = "#98FF98"
    BABY_BLUE = "#89CFF0"
    LILAC = "#C8A2C8"
    
    # Earth Tone
    OLIVE = "#808000"
    GREEN = "#008000"
    MOSS = "#8A9A5B"
    TERRA_COTTA = "#E2725B"
    BURGUNDY = "#800020"
    WINE = "#722F37"
    MUSTARD = "#FFDB58"


class ColorFamily(Enum):
    """Color families for matching"""
    NEUTRAL = "neutral"
    WARM = "warm"
    COOL = "cool"
    PASTEL = "pastel"
    EARTH = "earth"


class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"
    ALL_SEASON = "all"


class Occasion(Enum):
    CASUAL = "casual"
    WORK = "work"
    DATE = "date"
    PARTY = "party"
    SPORT = "sport"
    FORMAL = "formal"


class Style(Enum):
    CASUAL = "casual"
    FORMAL = "formal"
    TRENDY = "trendy"
    ATHLETIC = "athletic"
    MINIMALIST = "minimalist"
    PREPPY = "preppy"
    VINTAGE = "vintage"


@dataclass
class FashionItem:
    """Represents a fashion item with metadata"""
    item_id: str
    name: str
    category: str  # TOP, BOTTOM, FOOTWEAR, ACCESSORY
    subcategory: str  # TSHIRT, JEANS, etc.
    color: Color
    color_family: ColorFamily
    season: Season
    occasions: List[Occasion]
    styles: List[Style]
    formality_level: int  # 1-5, 5 being most formal
    warmth_level: int  # 1-5, 5 being warmest
    image_url: Optional[str] = None
    price: Optional[float] = None
    brand: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "name": self.name,
            "category": self.category,
            "subcategory": self.subcategory,
            "color": self.color.value,
            "color_family": self.color_family.value if self.color_family else None,
            "season": self.season.value,
            "occasions": [o.value for o in self.occasions],
            "styles": [s.value for s in self.styles],
            "formality_level": self.formality_level,
            "warmth_level": self.warmth_level,
            "image_url": self.image_url,
            "price": self.price,
            "brand": self.brand
        }


class ColorHarmonyRules:
    """
    Deterministic color matching rules.
    Uses fashion industry standard color theory.
    """
    
    COLOR_COMPATIBILITY = {
        ColorFamily.NEUTRAL: {
            ColorFamily.NEUTRAL: 1.0,
            ColorFamily.WARM: 0.9,
            ColorFamily.COOL: 0.9,
            ColorFamily.PASTEL: 0.95,
            ColorFamily.EARTH: 0.95,
        },
        ColorFamily.WARM: {
            ColorFamily.NEUTRAL: 0.9,
            ColorFamily.WARM: 0.85,
            ColorFamily.COOL: 0.6,
            ColorFamily.PASTEL: 0.9,
            ColorFamily.EARTH: 0.95,
        },
        ColorFamily.COOL: {
            ColorFamily.NEUTRAL: 0.9,
            ColorFamily.WARM: 0.6,
            ColorFamily.COOL: 0.85,
            ColorFamily.PASTEL: 0.9,
            ColorFamily.EARTH: 0.8,
        },
        ColorFamily.PASTEL: {
            ColorFamily.NEUTRAL: 0.95,
            ColorFamily.WARM: 0.9,
            ColorFamily.COOL: 0.9,
            ColorFamily.PASTEL: 0.8,
            ColorFamily.EARTH: 0.85,
        },
        ColorFamily.EARTH: {
            ColorFamily.NEUTRAL: 0.95,
            ColorFamily.WARM: 0.95,
            ColorFamily.COOL: 0.8,
            ColorFamily.PASTEL: 0.85,
            ColorFamily.EARTH: 0.9,
        },
    }
    
    CLASSIC_COMBINATIONS = [
        (Color.BLACK, Color.WHITE),
        (Color.BLACK, Color.GRAY),
        (Color.NAVY, Color.WHITE),
        (Color.BLACK, Color.BEIGE),
        (Color.BLUE, Color.TEAL),
        (Color.NAVY, Color.BLUE),
        (Color.GREEN, Color.OLIVE),
        (Color.BLUE, Color.ORANGE),
        (Color.RED, Color.ORANGE),
        (Color.PINK, Color.LAVENDER),
    ]
    
    @staticmethod
    def get_compatibility(color1: Color, color2: Color) -> float:
        family1 = ColorFamily.NEUTRAL
        family2 = ColorFamily.NEUTRAL
        
        # Determine color families based on value
        if color1 in [Color.WHITE, Color.BLACK, Color.GRAY, Color.NAVY, Color.BROWN, Color.BEIGE, Color.CAMEL]:
            family1 = ColorFamily.NEUTRAL
        elif color1 in [Color.RED, Color.ORANGE, Color.CORAL, Color.PEACH, Color.YELLOW, Color.GOLD]:
            family1 = ColorFamily.WARM
        elif color1 in [Color.BLUE, Color.TEAL, Color.TURQUOISE, Color.COBALT, Color.INDIGO]:
            family1 = ColorFamily.COOL
        elif color1 in [Color.PINK, Color.LAVENDER, Color.MINT, Color.BABY_BLUE, Color.LILAC]:
            family1 = ColorFamily.PASTEL
        elif color1 in [Color.OLIVE, Color.MOSS, Color.TERRA_COTTA, Color.BURGUNDY, Color.WINE, Color.MUSTARD]:
            family1 = ColorFamily.EARTH
            
        if color2 in [Color.WHITE, Color.BLACK, Color.GRAY, Color.NAVY, Color.BROWN, Color.BEIGE, Color.CAMEL]:
            family2 = ColorFamily.NEUTRAL
        elif color2 in [Color.RED, Color.ORANGE, Color.CORAL, Color.PEACH, Color.YELLOW, Color.GOLD]:
            family2 = ColorFamily.WARM
        elif color2 in [Color.BLUE, Color.TEAL, Color.TURQUOISE, Color.COBALT, Color.INDIGO]:
            family2 = ColorFamily.COOL
        elif color2 in [Color.PINK, Color.LAVENDER, Color.MINT, Color.BABY_BLUE, Color.LILAC]:
            family2 = ColorFamily.PASTEL
        elif color2 in [Color.OLIVE, Color.MOSS, Color.TERRA_COTTA, Color.BURGUNDY, Color.WINE, Color.MUSTARD]:
            family2 = ColorFamily.EARTH
        
        return ColorHarmonyRules.COLOR_COMPATIBILITY.get(family1, {}).get(family2, 0.7)
    
    @classmethod
    def is_classic_combo(cls, color1: Color, color2: Color) -> bool:
        combo = (color1, color2) if color1.value < color2.value else (color2, color1)
        return combo in ColorHarmonyRules.CLASSIC_COMBINATIONS


class StyleCompatibilityRules:
    STYLE_COMPATIBILITY = {
        Style.CASUAL: {Style.CASUAL: 1.0, Style.MINIMALIST: 0.9, Style.ATHLETIC: 0.85},
        Style.FORMAL: {Style.FORMAL: 1.0, Style.MINIMALIST: 0.9},
        Style.TRENDY: {Style.TRENDY: 1.0, Style.CASUAL: 0.8, Style.MINIMALIST: 0.75},
        Style.ATHLETIC: {Style.ATHLETIC: 1.0, Style.CASUAL: 0.9},
        Style.MINIMALIST: {Style.MINIMALIST: 1.0, Style.CASUAL: 0.9, Style.FORMAL: 0.85},
        Style.VINTAGE: {Style.VINTAGE: 1.0, Style.CASUAL: 0.8},
    }
    
    @staticmethod
    def get_style_score(style1: Style, style2: Style) -> float:
        return StyleCompatibilityRules.STYLE_COMPATIBILITY.get(style1, {}).get(style2, 0.5)
    
    @staticmethod
    def formality_gap_okay(f1: int, f2: int) -> bool:
        gap = abs(f1 - f2)
        return gap <= 2


class SeasonalRules:
    SEASONAL_APPROPRIATENESS = {
        Season.ALL_SEASON: [1.0, 1.0, 1.0, 1.0],
        Season.SPRING: [0.9, 0.7, 0.5, 0.3],
        Season.SUMMER: [0.5, 1.0, 0.3, 0.2],
        Season.FALL: [0.3, 0.5, 0.9, 0.7],
        Season.WINTER: [0.2, 0.3, 0.7, 1.0],
    }
    
    LAYER_RULES = {
        Season.SPRING: {"base": 1, "layers": 2, "outer": "light"},
        Season.SUMMER: {"base": 1, "layers": 0, "outer": None},
        Season.FALL: {"base": 1, "layers": 2, "outer": "medium"},
        Season.WINTER: {"base": 2, "layers": 3, "outer": "heavy"},
    }
    
    @staticmethod
    def get_seasonal_score(item_season: Season, target_season: Season) -> float:
        season_scores = SeasonalRules.SEASONAL_APPROPRIATENESS.get(item_season, [0.5] * 4)
        season_idx = list(Season).index(target_season)
        return season_scores[season_idx] if season_idx < len(season_scores) else 0.5


class OutfitAssembler:
    def __init__(self):
        self.color_rules = ColorHarmonyRules()
        self.style_rules = StyleCompatibilityRules()
        self.seasonal_rules = SeasonalRules()
        self.items: Dict[str, FashionItem] = {}
    
    def add_item(self, item: FashionItem):
        self.items[item.item_id] = item
    
    def assemble_outfit(
        self,
        season: Season,
        occasion: Occasion,
        top_id: Optional[str] = None,
        bottom_id: Optional[str] = None,
        footwear_id: Optional[str] = None,
        accessory_id: Optional[str] = None,
        style_preference: Optional[Style] = None,
    ) -> Tuple[Optional[List[FashionItem]], float]:
        outfit = []
        score = 1.0
        
        available = self._filter_items(season, occasion, style_preference)
        
        if top_id and top_id in available.get("TOP", {}):
            top = self.items[top_id]
        else:
            top = self._select_best_item(available.get("TOP", {}), season, occasion)
        
        if top:
            outfit.append(top)
        
        if bottom_id and bottom_id in available.get("BOTTOM", {}):
            bottom = self.items[bottom_id]
        else:
            bottom = self._select_best_item(available.get("BOTTOM", {}), season, occasion)
        
        if bottom:
            outfit.append(bottom)
            if top:
                score *= self._check_pairing(top, bottom)
        
        if footwear_id and footwear_id in available.get("FOOTWEAR", {}):
            footwear = self.items[footwear_id]
        else:
            footwear = self._select_best_item(available.get("FOOTWEAR", {}), season, occasion)
        
        if footwear:
            outfit.append(footwear)
            if bottom:
                score *= self._check_pairing(bottom, footwear)
        
        if accessory_id and accessory_id in available.get("ACCESSORIES", {}):
            accessory = self.items[accessory_id]
        elif accessory_id is None:
            accessory = self._select_best_item(available.get("ACCESSORIES", {}), season, occasion)
        else:
            accessory = None
        
        if accessory:
            outfit.append(accessory)
        
        return (outfit if outfit else None, score)
    
    def _filter_items(
        self,
        season: Season,
        occasion: Occasion,
        style_preference: Optional[Style] = None
    ) -> Dict[str, Dict[str, FashionItem]]:
        filtered = {"TOP": {}, "BOTTOM": {}, "FOOTWEAR": {}, "ACCESSORIES": {}}
        
        for item_id, item in self.items.items():
            season_score = self.seasonal_rules.get_seasonal_score(item.season, season)
            if season_score < 0.3:
                continue
            
            if occasion not in item.occasions:
                continue
            
            if style_preference and style_preference not in item.styles:
                continue
            
            category = item.category
            if category in filtered:
                filtered[category][item_id] = item
        
        return filtered
    
    def _select_best_item(
        self,
        items: Dict[str, FashionItem],
        season: Season,
        occasion: Occasion
    ) -> Optional[FashionItem]:
        if not items:
            return None
        
        scored = []
        for item_id, item in items.items():
            season_score = self.seasonal_rules.get_seasonal_score(item.season, season)
            occasion_score = 1.0 if occasion in item.occasions else 0.5
            formality_score = 1.0 if occasion.value in ["formal", "work"] else 0.9
            
            total_score = season_score * occasion_score * formality_score
            scored.append((total_score, item))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[0][1] if scored else None
    
    def _check_pairing(self, item1: FashionItem, item2: FashionItem) -> float:
        score = 1.0
        
        color_score = self.color_rules.get_compatibility(item1.color, item2.color)
        score *= color_score
        
        best_style_score = 0
        for s1 in item1.styles:
            for s2 in item2.styles:
                style_score = self.style_rules.get_style_score(s1, s2)
                if style_score > best_style_score:
                    best_style_score = style_score
        score *= (best_style_score if best_style_score else 0.8)
        
        if self.style_rules.formality_gap_okay(item1.formality_level, item2.formality_level):
            score *= 0.95
        else:
            score *= 0.7
        
        return score
    
    def generate_outfit_set(
        self,
        season: Season,
        occasion: Occasion,
        style_preference: Optional[Style] = None,
        include_accessories: bool = True,
        include_outerwear: bool = True,
    ) -> Dict:
        outfit, score = self.assemble_outfit(
            season=season,
            occasion=occasion,
            style_preference=style_preference,
        )
        
        if not outfit:
            return {"error": "No suitable items found"}
        
        outfit_dict = {
            "outfit_id": f"OUTFIT_{season.value}_{occasion.value}",
            "season": season.value,
            "occasion": occasion.value,
            "style": style_preference.value if style_preference else "mixed",
            "compatibility_score": round(score, 2),
            "layers": {
                "base_layer": None,
                "middle_layer": None,
                "outerwear": None,
                "bottom": None,
                "footwear": None,
                "accessories": []
            },
            "items": [],
            "color_palette": [],
            "style_tags": [],
            "recommendations": []
        }
        
        for item in outfit:
            item_dict = item.to_dict()
            item_dict["score"] = self._calculate_item_score(item, season, occasion)
            outfit_dict["items"].append(item_dict)
            
            outfit_dict["color_palette"].append(item.color.value)
            
            for style in item.styles:
                if style.value not in outfit_dict["style_tags"]:
                    outfit_dict["style_tags"].append(style.value)
            
            if item.category == "TOP" and item.warmth_level <= 2:
                outfit_dict["layers"]["base_layer"] = item_dict
            elif item.category == "TOP" and item.warmth_level >= 3:
                outfit_dict["layers"]["outerwear"] = item_dict
            elif item.category == "BOTTOM":
                outfit_dict["layers"]["bottom"] = item_dict
            elif item.category == "FOOTWEAR":
                outfit_dict["layers"]["footwear"] = item_dict
            elif item.category == "ACCESSORIES":
                outfit_dict["layers"]["accessories"].append(item_dict)
        
        outfit_dict["recommendations"] = self._generate_recommendations(outfit_dict, season, occasion)
        
        return outfit_dict
    
    def _calculate_item_score(self, item: FashionItem, season: Season, occasion: Occasion) -> float:
        season_score = self.seasonal_rules.get_seasonal_score(item.season, season)
        occasion_score = 1.0 if occasion in item.occasions else 0.5
        return round(season_score * occasion_score, 2)
    
    def _generate_recommendations(self, outfit: Dict, season: Season, occasion: Occasion) -> List[str]:
        recs = []
        
        if season == Season.SPRING:
            recs.append("Perfect for spring! Light layers work beautifully.")
        elif season == Season.SUMMER:
            recs.append("Stay cool with this lightweight combination.")
        elif season == Season.FALL:
            recs.append("Great fall layering - practical and stylish!")
        elif season == Season.WINTER:
            recs.append("Keep warm with these cold-weather essentials.")
        
        if occasion == Occasion.CASUAL:
            recs.append("Easy, relaxed vibes for everyday wear.")
        elif occasion == Occasion.WORK:
            recs.append("Professional yet approachable - great for the office.")
        elif occasion == Occasion.DATE:
            recs.append("Put-together yet approachable - perfect for a date!")
        
        recs.append("Color palette is harmoniously coordinated.")
        
        return recs


# Sample Database
SAMPLE_DATABASE = [
    FashionItem(
        item_id="TOP001", name="Classic White Tee", category="TOP", subcategory="TSHIRT",
        color=Color.WHITE, color_family=ColorFamily.NEUTRAL, season=Season.ALL_SEASON,
        occasions=[Occasion.CASUAL, Occasion.DATE], styles=[Style.CASUAL, Style.MINIMALIST],
        formality_level=1, warmth_level=1, image_url="https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400"
    ),
    FashionItem(
        item_id="TOP002", name="Navy Oxford Shirt", category="TOP", subcategory="SHIRT",
        color=Color.NAVY, color_family=ColorFamily.NEUTRAL, season=Season.ALL_SEASON,
        occasions=[Occasion.WORK, Occasion.FORMAL, Occasion.DATE], styles=[Style.FORMAL, Style.MINIMALIST],
        formality_level=4, warmth_level=2, image_url="https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400"
    ),
    FashionItem(
        item_id="TOP003", name="Beige Cable Knit Sweater", category="TOP", subcategory="SWEATER",
        color=Color.BEIGE, color_family=ColorFamily.EARTH, season=Season.FALL,
        occasions=[Occasion.CASUAL, Occasion.DATE], styles=[Style.CASUAL, Style.VINTAGE],
        formality_level=2, warmth_level=4, image_url="https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=400"
    ),
    FashionItem(
        item_id="TOP004", name="Black Leather Jacket", category="TOP", subcategory="JACKET",
        color=Color.BLACK, color_family=ColorFamily.NEUTRAL, season=Season.ALL_SEASON,
        occasions=[Occasion.CASUAL, Occasion.DATE, Occasion.PARTY], styles=[Style.TRENDY, Style.CASUAL],
        formality_level=3, warmth_level=3, image_url="https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400"
    ),
    FashionItem(
        item_id="BOTTOM001", name="Blue Straight Jeans", category="BOTTOM", subcategory="JEANS",
        color=Color.BLUE, color_family=ColorFamily.COOL, season=Season.ALL_SEASON,
        occasions=[Occasion.CASUAL, Occasion.DATE], styles=[Style.CASUAL],
        formality_level=2, warmth_level=2, image_url="https://images.unsplash.com/photo-1542272604-787c3835535d?w=400"
    ),
    FashionItem(
        item_id="BOTTOM002", name="Black Dress Pants", category="BOTTOM", subcategory="TROUSERS",
        color=Color.BLACK, color_family=ColorFamily.NEUTRAL, season=Season.ALL_SEASON,
        occasions=[Occasion.WORK, Occasion.FORMAL, Occasion.DATE], styles=[Style.FORMAL],
        formality_level=5, warmth_level=1, image_url="https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=400"
    ),
    FashionItem(
        item_id="BOTTOM003", name="Beige Chinos", category="BOTTOM", subcategory="CHINOS",
        color=Color.CAMEL, color_family=ColorFamily.EARTH, season=Season.SPRING,
        occasions=[Occasion.CASUAL, Occasion.WORK, Occasion.DATE], styles=[Style.CASUAL, Style.PREPPY],
        formality_level=3, warmth_level=2, image_url="https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=400"
    ),
    FashionItem(
        item_id="FOOTWEAR001", name="White Low-Top Sneakers", category="FOOTWEAR", subcategory="SNEAKERS",
        color=Color.WHITE, color_family=ColorFamily.NEUTRAL, season=Season.ALL_SEASON,
        occasions=[Occasion.CASUAL, Occasion.SPORT], styles=[Style.CASUAL, Style.ATHLETIC],
        formality_level=1, warmth_level=1, image_url="https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400"
    ),
    FashionItem(
        item_id="FOOTWEAR002", name="Brown Leather Loafers", category="FOOTWEAR", subcategory="LOAFERS",
        color=Color.BROWN, color_family=ColorFamily.EARTH, season=Season.ALL_SEASON,
        occasions=[Occasion.WORK, Occasion.FORMAL, Occasion.DATE], styles=[Style.FORMAL, Style.MINIMALIST],
        formality_level=4, warmth_level=2, image_url="https://images.unsplash.com/photo-1614252369475-531eba835eb1?w=400"
    ),
    FashionItem(
        item_id="FOOTWEAR003", name="Black Chelsea Boots", category="FOOTWEAR", subcategory="BOOTS",
        color=Color.BLACK, color_family=ColorFamily.NEUTRAL, season=Season.FALL,
        occasions=[Occasion.CASUAL, Occasion.DATE], styles=[Style.TRENDY, Style.CASUAL],
        formality_level=3, warmth_level=3, image_url="https://images.unsplash.com/photo-1608256246200-53e635b5b65f?w=400"
    ),
    FashionItem(
        item_id="ACC001", name="Silver Chain Necklace", category="ACCESSORIES", subcategory="NECKLACE",
        color=Color.GRAY, color_family=ColorFamily.NEUTRAL, season=Season.ALL_SEASON,
        occasions=[Occasion.DATE, Occasion.PARTY, Occasion.CASUAL], styles=[Style.TRENDY, Style.MINIMALIST],
        formality_level=2, warmth_level=1, image_url="https://images.unsplash.com/photo-1599643478518-a784e5dc4c8f?w=400"
    ),
    FashionItem(
        item_id="ACC002", name="Leather Belt", category="ACCESSORIES", subcategory="BELT",
        color=Color.BROWN, color_family=ColorFamily.EARTH, season=Season.ALL_SEASON,
        occasions=[Occasion.WORK, Occasion.FORMAL, Occasion.CASUAL], styles=[Style.FORMAL, Style.CASUAL],
        formality_level=3, warmth_level=1, image_url="https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400"
    ),
]


if __name__ == "__main__":
    assembler = OutfitAssembler()
    
    for item in SAMPLE_DATABASE:
        assembler.add_item(item)
    
    outfit = assembler.generate_outfit_set(
        season=Season.SPRING,
        occasion=Occasion.CASUAL,
        style_preference=Style.CASUAL
    )
    
    print("\n🎨 GENERATED OUTFIT SET")
    print("=" * 50)
    print(json.dumps(outfit, indent=2))

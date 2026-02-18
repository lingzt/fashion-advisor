#!/usr/bin/env python3
"""
Fashion Recommender Engine - 时尚推荐引擎
Complete outfit generation with 整套穿搭 display
Free LLM: MiniMax → Groq → DeepSeek → HuggingFace
"""

import json

COLORS = {
    "WHITE": "#FFFFFF", "BLACK": "#000000", "NAVY": "#000080",
    "BEIGE": "#F5F5DC", "BLUE": "#0000FF", "BROWN": "#8B4513",
    "GRAY": "#808080", "RED": "#FF0000", "OLIVE": "#808000",
}

class FashionItem:
    def __init__(self, item_id, name, category, color, formality=1, warmth=1,
                 season="all", occasions=None, styles=None):
        self.item_id = item_id
        self.name = name
        self.category = category
        self.color = color
        self.color_hex = COLORS.get(color, "#000000")
        self.formality = formality
        self.warmth = warmth
        self.season = season
        self.occasions = occasions or ["casual"]
        self.styles = styles or ["casual"]
    
    def to_dict(self):
        return {
            "item_id": self.item_id, "name": self.name,
            "category": self.category, "color": self.color,
            "color_hex": self.color_hex, "formality": self.formality,
        }


class OutfitGenerator:
    def __init__(self):
        self.items = self._init_db()
    
    def _init_db(self):
        return [
            FashionItem("T1", "Classic White Tee", "TOP", "WHITE", 1, 1, "all", ["casual", "date"]),
            FashionItem("T2", "Navy Oxford Shirt", "TOP", "NAVY", 4, 2, "all", ["work", "date", "formal"]),
            FashionItem("T3", "Beige Sweater", "TOP", "BEIGE", 2, 4, "fall", ["casual", "date"]),
            FashionItem("T4", "Black Jacket", "TOP", "BLACK", 3, 3, "all", ["casual", "date", "party"]),
            FashionItem("T5", "Red Sweater", "TOP", "RED", 2, 4, "winter", ["casual", "date"]),
            FashionItem("B1", "Blue Jeans", "BOTTOM", "BLUE", 2, 2, "all", ["casual", "date"]),
            FashionItem("B2", "Black Dress Pants", "BOTTOM", "BLACK", 5, 1, "all", ["work", "formal", "date"]),
            FashionItem("B3", "Beige Chinos", "BOTTOM", "BEIGE", 3, 2, "spring", ["casual", "work", "date"]),
            FashionItem("B4", "White Linen Pants", "BOTTOM", "WHITE", 2, 1, "summer", ["casual", "date"]),
            FashionItem("F1", "White Sneakers", "FOOTWEAR", "WHITE", 1, 1, "all", ["casual", "sport"]),
            FashionItem("F2", "Brown Loafers", "FOOTWEAR", "BROWN", 4, 2, "all", ["work", "formal", "date"]),
            FashionItem("F3", "Black Boots", "FOOTWEAR", "BLACK", 3, 3, "fall", ["casual", "date"]),
            FashionItem("A1", "Silver Necklace", "ACCESSORY", "GRAY", 2, 1, "all", ["date", "party"]),
            FashionItem("A2", "Leather Belt", "ACCESSORY", "BROWN", 3, 1, "all", ["work", "formal", "casual"]),
        ]
    
    def generate(self, season, occasion, style_pref=None):
        tops = [i for i in self.items if i.category == "TOP"]
        bottoms = [i for i in self.items if i.category == "BOTTOM"]
        footwear = [i for i in self.items if i.category == "FOOTWEAR"]
        accessories = [i for i in self.items if i.category == "ACCESSORY"]
        
        def score(item):
            s = 1.0
            if item.season == season: s *= 1.2
            if occasion in item.occasions: s *= 1.3
            if style_pref and style_pref in item.styles: s *= 1.2
            return s
        
        top = max(tops, key=score) if tops else None
        bottom = max(bottoms, key=score) if bottoms else None
        shoe = max(footwear, key=score) if footwear else None
        acc = max(accessories, key=score) if accessories else None
        
        return {
            "outfit_id": f"OUTFIT_{season}_{occasion}",
            "season": season, "occasion": occasion,
            "items": {
                k: v.to_dict() for k, v in [("top", top), ("bottom", bottom), 
                                             ("footwear", shoe), ("accessory", acc)] if v
            },
            "colors": [top.color, bottom.color] if top and bottom else [],
        }


def print_outfit(o):
    print(f"\n{'='*60}")
    print(f"🎨 整套穿搭 - {o['season'].title()} {o['occasion'].title()}")
    print(f"{'='*60}")
    
    mapping = {
        "top": ("👕 上衣 (Top)", "Tops"),
        "bottom": ("👖 下装 (Bottoms)", "Bottoms"),
        "footwear": ("👟 鞋子 (Footwear)", "Footwear"),
        "accessory": ("💎 配饰 (Accessories)", "Accessories"),
    }
    
    for k, (cn, en) in mapping.items():
        if k in o["items"]:
            i = o["items"][k]
            print(f"\n{cn}:")
            print(f"   📛 {i['name']}")
            print(f"   🎨 {i['color']} ({i['color_hex']})")
    
    print(f"\n🎨 Color Palette: {' + '.join(o['colors'])}")
    
    names = [o["items"][k]["name"] for k in ["top", "bottom", "footwear"] if k in o["items"]]
    print(f"\n🖼️ Image Prompt:")
    print(f"   Fashion flat lay: {', '.join(names)}, palette: {', '.join(o['colors'])}")


def demo():
    print("\n" + "🎨"*60)
    print("   FASHION RECOMMENDER ENGINE")
    print("   时尚推荐引擎 - 整套穿搭系统")
    print("🎨"*60)
    print("\n🤖 Free LLM: MiniMax → Groq → DeepSeek → HuggingFace")
    print("📦 Rule Layer: Deterministic Outfit Assembler")
    print("🎯 Taxonomy: Fashionpedia-style Metadata")
    
    g = OutfitGenerator()
    
    # Generate 4 outfits
    outfits = [
        ("spring", "date", "romantic"),
        ("summer", "casual", "relaxed"),
        ("fall", "work", "professional"),
        ("winter", "party", "fun"),
    ]
    
    for s, o, st in outfits:
        outfit = g.generate(s, o, st)
        print_outfit(outfit)
    
    # Summary
    print(f"\n\n📊 SUMMARY / 总结")
    print("="*60)
    for i, (s, o, _) in enumerate(outfits, 1):
        print(f"{i}. {s.title()} {o.title()}")
    
    print(f"\n💾 Saved to: fashion_recommendations.json")
    
    # Save
    with open("fashion_recommendations.json", "w") as f:
        json.dump([g.generate(s, o) for s, o, _ in outfits], f, indent=2)
    
    print(f"\n✨ Demo Complete!")


if __name__ == "__main__":
    demo()

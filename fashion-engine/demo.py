#!/usr/bin/env python3
"""
Fashion Recommender Demo
=======================
Complete demo showing outfit generation, image prompt,
and整套穿搭 (complete outfit set) display.
"""

import json
from rules.outfit_assembler import (
    OutfitAssembler, Season, Occasion, Style,
    FashionItem, Color
)
from src.chatbot import FashionChatbot


def print_fashion_image(image_url: str, title: str = "Outfit Image"):
    """Display fashion image (placeholder for actual image display)"""
    print(f"\n{'='*60}")
    print(f"🖼️  {title}")
    print(f"{'='*60}")
    print(f"   📱 Image URL: {image_url}")
    print(f"   💡 Use this URL to view the outfit image")
    print(f"   📌 Or generate with DALL-E 3:")
    print(f"   🎨 Prompt: (see below)")
    print(f"{'='*60}\n")


def demo_complete_outfit():
    """Demo complete outfit generation with image"""
    
    print("\n" + "🎨"*30)
    print("   FASHION RECOMMENDER ENGINE DEMO")
    print("   整套穿搭推荐系统演示")
    print("🎨"*30 + "\n")
    
    # Initialize
    assembler = OutfitAssembler()
    
    # Add sample items (would come from database in production)
    sample_items = [
        FashionItem(
            item_id="TOP001", name="Classic White Tee", category="TOP", subcategory="TSHIRT",
            color=Color.WHITE, color_family=None, season=Season.ALL_SEASON,
            occasions=[Occasion.CASUAL, Occasion.DATE], styles=[Style.CASUAL],
            formality_level=1, warmth_level=1
        ),
        FashionItem(
            item_id="TOP004", name="Black Leather Jacket", category="TOP", subcategory="JACKET",
            color=Color.BLACK, color_family=None, season=Season.ALL_SEASON,
            occasions=[Occasion.CASUAL, Occasion.DATE], styles=[Style.TRENDY],
            formality_level=3, warmth_level=3
        ),
        FashionItem(
            item_id="BOTTOM001", name="Blue Straight Jeans", category="BOTTOM", subcategory="JEANS",
            color=Color.BLUE, color_family=None, season=Season.ALL_SEASON,
            occasions=[Occasion.CASUAL, Occasion.DATE], styles=[Style.CASUAL],
            formality_level=2, warmth_level=2
        ),
        FashionItem(
            item_id="FOOTWEAR001", name="White Sneakers", category="FOOTWEAR", subcategory="SNEAKERS",
            color=Color.WHITE, color_family=None, season=Season.ALL_SEASON,
            occasions=[Occasion.CASUAL], styles=[Style.CASUAL],
            formality_level=1, warmth_level=1
        ),
        FashionItem(
            item_id="ACC001", name="Silver Necklace", category="ACCESSORIES", subcategory="NECKLACE",
            color=Color.GRAY, color_family=None, season=Season.ALL_SEASON,
            occasions=[Occasion.DATE, Occasion.PARTY], styles=[Style.TRENDY],
            formality_level=2, warmth_level=1
        ),
    ]
    
    for item in sample_items:
        assembler.add_item(item)
    
    # Generate outfit
    outfit = assembler.generate_outfit_set(
        season=Season.SPRING,
        occasion=Occasion.DATE,
        style_preference=Style.CASUAL
    )
    
    # Display outfit
    print("🌸 SPRING DATE NIGHT OUTFIT")
    print("=" * 60)
    print(f"📅 Season: {outfit['season'].title()}")
    print(f"🎯 Occasion: {outfit['occasion'].title()}")
    print(f"✨ Style: {outfit['style'].title()}")
    print(f"📊 Compatibility Score: {outfit['compatibility_score']}")
    
    print("\n👗 整套穿搭 (Complete Outfit Set):")
    print("-" * 60)
    
    layers = outfit['layers']
    
    if layers['base_layer']:
        item = layers['base_layer']
        print(f"👕 上衣 (Top): {item['name']}")
        print(f"   🎨 Color: {item['color']}")
    
    if layers['outerwear']:
        item = layers['outerwear']
        print(f"🧥 外套 (Outerwear): {item['name']}")
        print(f"   🎨 Color: {item['color']}")
    
    if layers['bottom']:
        item = layers['bottom']
        print(f"👖 下装 (Bottom): {item['name']}")
        print(f"   🎨 Color: {item['color']}")
    
    if layers['footwear']:
        item = layers['footwear']
        print(f"👟 鞋子 (Footwear): {item['name']}")
        print(f"   🎨 Color: {item['color']}")
    
    if layers['accessories']:
        print("\n💎 配饰 (Accessories):")
        for acc in layers['accessories']:
            print(f"   • {acc['name']} ({acc['color']})")
    
    # Color palette
    print(f"\n🎨 配色方案 (Color Palette):")
    print(f"   {' + '.join(outfit['color_palette'])}")
    
    # Style tags
    print(f"\n🏷️ 风格标签 (Style Tags):")
    print(f"   {', '.join(outfit['style_tags'])}")
    
    # Recommendations
    print(f"\n💡 搭配建议 (Style Notes):")
    for rec in outfit['recommendations']:
        print(f"   • {rec}")
    
    # Generate image prompt
    chatbot = FashionChatbot()
    image_prompt = chatbot.generate_outfit_image_prompt(outfit)
    
    print(f"\n🖼️  生成图片 (Image Generation):")
    print(f"   Prompt for DALL-E 3 / Midjourney:")
    print(f"   ─────────────────────────────────────────")
    print(f"   {image_prompt}")
    print(f"   ─────────────────────────────────────────\n")
    
    # Save outfit
    output_file = "generated_outfit.json"
    with open(output_file, 'w') as f:
        json.dump(outfit, f, indent=2)
    print(f"💾 Outfit saved to: {output_file}\n")
    
    return outfit


def demo_chatbot():
    """Demo chatbot interactions"""
    
    print("\n" + "💬"*30)
    print("   CHATBOT INTERACTIONS")
    print("💬"*30 + "\n")
    
    chatbot = FashionChatbot()
    
    # Test conversations
    conversations = [
        ("Recommend an outfit for a date tonight", "💕 Date Night"),
        ("What should I wear for a casual spring day?", "🌸 Spring Casual"),
        ("Generate a trendy outfit", "🔥 Trendy Look"),
    ]
    
    for user_msg, context in conversations:
        print(f"\n👤 User: {user_msg}")
        print("-" * 50)
        response = chatbot.chat(user_msg)
        print(f"🤖 Bot: {response[:200]}..." if len(response) > 200 else f"🤖 Bot: {response}")
        print()


def demo_all_seasons():
    """Demo outfits for all seasons"""
    
    print("\n" + "🌈"*30)
    print("   SEASONAL OUTFIT COLLECTION")
    print("🌈"*30 + "\n")
    
    assembler = OutfitAssembler()
    
    # Add items
    items = [
        FashionItem(item_id="t1", name="White Tee", category="TOP", subcategory="TSHIRT",
                   color=Color.WHITE, color_family=None, season=Season.ALL_SEASON,
                   occasions=[Occasion.CASUAL], styles=[Style.CASUAL],
                   formality_level=1, warmth_level=1),
        FashionItem(item_id="t2", name="Sweater", category="TOP", subcategory="SWEATER",
                   color=Color.BEIGE, color_family=None, season=Season.FALL,
                   occasions=[Occasion.CASUAL], styles=[Style.CASUAL],
                   formality_level=2, warmth_level=4),
        FashionItem(item_id="t3", name="Blazer", category="TOP", subcategory="JACKET",
                   color=Color.NAVY, color_family=None, season=Season.ALL_SEASON,
                   occasions=[Occasion.WORK, Occasion.DATE], styles=[Style.FORMAL],
                   formality_level=4, warmth_level=2),
        FashionItem(item_id="b1", name="Jeans", category="BOTTOM", subcategory="JEANS",
                   color=Color.BLUE, color_family=None, season=Season.ALL_SEASON,
                   occasions=[Occasion.CASUAL], styles=[Style.CASUAL],
                   formality_level=2, warmth_level=2),
        FashionItem(item_id="b2", name="Wool Pants", category="BOTTOM", subcategory="TROUSERS",
                   color=Color.BLACK, color_family=None, season=Season.WINTER,
                   occasions=[Occasion.WORK, Occasion.FORMAL], styles=[Style.FORMAL],
                   formality_level=4, warmth_level=3),
        FashionItem(item_id="f1", name="Sneakers", category="FOOTWEAR", subcategory="SNEAKERS",
                   color=Color.WHITE, color_family=None, season=Season.ALL_SEASON,
                   occasions=[Occasion.CASUAL], styles=[Style.CASUAL],
                   formality_level=1, warmth_level=1),
        FashionItem(item_id="f2", name="Boots", category="FOOTWEAR", subcategory="BOOTS",
                   color=Color.BLACK, color_family=None, season=Season.FALL,
                   occasions=[Occasion.CASUAL], styles=[Style.TRENDY],
                   formality_level=3, warmth_level=3),
    ]
    
    for item in items:
        assembler.add_item(item)
    
    # Generate outfits for each season
    seasons = [
        (Season.SPRING, "🌸 Spring", Occasion.CASUAL, Style.CASUAL),
        (Season.SUMMER, "☀️ Summer", Occasion.CASUAL, Style.CASUAL),
        (Season.FALL, "🍂 Fall", Occasion.CASUAL, Style.CASUAL),
        (Season.WINTER, "❄️ Winter", Occasion.CASUAL, Style.CASUAL),
    ]
    
    for season, season_name, occasion, style in seasons:
        outfit = assembler.generate_outfit_set(
            season=season,
            occasion=occasion,
            style_preference=style,
        )
        
        print(f"\n{season_name}:")
        print(f"   {outfit['outfit_id']}")
        print(f"   🎨 Colors: {', '.join(outfit['color_palette'][:3])}")
        print(f"   📊 Score: {outfit['compatibility_score']}")


if __name__ == "__main__":
    # Run demos
    demo_complete_outfit()
    demo_chatbot()
    demo_all_seasons()
    
    print("\n" + "✨"*30)
    print("   Demo Complete!")
    print("✨"*30 + "\n")

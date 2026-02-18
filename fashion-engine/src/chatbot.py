"""
Fashion Recommender Chatbot
===========================
LLM-powered fashion recommendation chatbot with OpenAI integration.
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# OpenAI imports (optional - graceful fallback)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from rules.outfit_assembler import (
    OutfitAssembler, Season, Occasion, Style, 
    FashionItem, SAMPLE_DATABASE
)


class UserPreference(Enum):
    """User preference categories"""
    CASUAL_WEAR = "casual_wear"
    FORMAL_WEAR = "formal_wear"
    STREETWEAR = "streetwear"
    MINIMALIST = "minimalist"
    VINTAGE = "vintage"
    ATHLETIC = "athletic"


@dataclass
class UserProfile:
    """User profile with preferences"""
    name: str
    preferred_styles: List[Style]
    preferred_colors: List[str]
    size: Optional[str] = None
    budget_range: Optional[tuple] = None  # (min, max)
    favorite_brands: List[str] = None
    avoid_colors: List[str] = None
    climate: Optional[str] = None  # tropical, temperate, continental


class FashionChatbot:
    """
    LLM-powered fashion recommendation chatbot.
    Uses OpenAI for natural language understanding and
    deterministic rules for outfit generation.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.assembler = OutfitAssembler()
        self.client = None
        self.user_profile = None
        self.conversation_history: List[Dict] = []
        
        # Initialize with sample database
        for item in SAMPLE_DATABASE:
            self.assembler.add_item(item)
        
        # Initialize OpenAI client if key provided
        if api_key and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=api_key)
        elif not OPENAI_AVAILABLE:
            print("⚠️  OpenAI not installed. Running in rule-only mode.")
            print("   Install with: pip install openai")
    
    def set_user_profile(self, profile: UserProfile):
        """Set user profile for personalized recommendations"""
        self.user_profile = profile
    
    def chat(self, message: str) -> str:
        """
        Main chat interface.
        Parses user intent and generates responses.
        """
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Detect intent
        intent = self._detect_intent(message)
        
        # Generate response based on intent
        response = self._handle_intent(intent, message)
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _detect_intent(self, message: str) -> str:
        """Detect user intent using simple keyword matching"""
        message_lower = message.lower()
        
        # Recommendation intents
        if any(word in message_lower for word in ["recommend", "suggest", "outfit", "what to wear", "what should i wear"]):
            return "recommendation"
        
        # Search intents
        if any(word in message_lower for word in ["find", "search", "look for", "do you have"]):
            return "search"
        
        # Outfit generation intents
        if any(word in message_lower for word in ["put together", "assemble", "create outfit", "generate outfit"]):
            return "outfit_generation"
        
        # Style advice intents
        if any(word in message_lower for word in ["style advice", "how to style", "how do i style", "what goes with"]):
            return "style_advice"
        
        # Feedback intents
        if any(word in message_lower for word in ["like", "love", "great", "perfect", "not my style", "don't like"]):
            return "feedback"
        
        # Help intents
        if any(word in message_lower for word in ["help", "what can you do", "capabilities"]):
            return "help"
        
        # Greeting
        if any(word in message_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return "greeting"
        
        return "general"
    
    def _handle_intent(self, intent: str, message: str) -> str:
        """Handle detected intent"""
        handlers = {
            "recommendation": self._handle_recommendation,
            "search": self._handle_search,
            "outfit_generation": self._handle_outfit_generation,
            "style_advice": self._handle_style_advice,
            "feedback": self._handle_feedback,
            "help": self._handle_help,
            "greeting": self._handle_greeting,
            "general": self._handle_general,
        }
        
        handler = handlers.get(intent, self._handle_general)
        return handler(message)
    
    def _parse_context(self, message: str) -> Dict:
        """Parse season, occasion, and style from message"""
        context = {
            "season": None,
            "occasion": None,
            "style": None,
            "colors": []
        }
        
        message_lower = message.lower()
        
        # Season detection
        seasons = {
            "spring": Season.SPRING,
            "summer": Season.SUMMER,
            "fall": Season.FALL,
            "autumn": Season.FALL,
            "winter": Season.WINTER,
        }
        for word, season in seasons.items():
            if word in message_lower:
                context["season"] = season
                break
        
        # Occasion detection
        occasions = {
            "casual": Occasion.CASUAL,
            "work": Occasion.WORK,
            "office": Occasion.WORK,
            "date": Occasion.DATE,
            "party": Occasion.PARTY,
            "formal": Occasion.FORMAL,
            "business": Occasion.WORK,
            "sport": Occasion.SPORT,
            "gym": Occasion.SPORT,
        }
        for word, occasion in occasions.items():
            if word in message_lower:
                context["occasion"] = occasion
                break
        
        # Style detection
        styles = {
            "casual": Style.CASUAL,
            "formal": Style.FORMAL,
            "trendy": Style.TRENDY,
            "athletic": Style.ATHLETIC,
            "minimalist": Style.MINIMALIST,
            "vintage": Style.VINTAGE,
        }
        for word, style in styles.items():
            if word in message_lower:
                context["style"] = style
                break
        
        return context
    
    def _handle_recommendation(self, message: str) -> str:
        """Handle outfit recommendations"""
        context = self._parse_context(message)
        
        # Use user preferences if set
        season = context.get("season") or Season.ALL_SEASON
        occasion = context.get("occasion") or Occasion.CASUAL
        style = context.get("style")
        
        # Generate outfit
        outfit = self.assembler.generate_outfit_set(
            season=season,
            occasion=occasion,
            style_preference=style,
        )
        
        return self._format_outfit_response(outfit)
    
    def _handle_outfit_generation(self, message: str) -> str:
        """Handle outfit generation requests"""
        context = self._parse_context(message)
        
        season = context.get("season") or Season.SPRING
        occasion = context.get("occasion") or Occasion.CASUAL
        style = context.get("style")
        
        outfit = self.assembler.generate_outfit_set(
            season=season,
            occasion=occasion,
            style_preference=style,
            include_accessories=True,
        )
        
        response = f"✨ {self._generate_outfit_title(outfit)}\n\n"
        response += self._format_outfit_items(outfit)
        response += self._format_recommendations(outfit)
        
        return response
    
    def _handle_search(self, message: str) -> str:
        """Handle product search"""
        message_lower = message.lower()
        
        # Search in item names and categories
        results = []
        for item_id, item in self.assembler.items.items():
            if (message_lower in item.name.lower() or 
                message_lower in item.category.lower() or
                message_lower in item.subcategory.lower()):
                results.append(item)
        
        if not results:
            return "I couldn't find any items matching your search. Try searching for specific items like 'jeans', 'shirts', or 'sneakers'."
        
        response = f"🔍 Found {len(results)} items:\n\n"
        for item in results[:5]:  # Limit to 5 results
            response += f"• {item.name} - {item.color.value} {item.category}\n"
            response += f"  Style: {', '.join(s.value for s in item.styles)}\n"
            response += f"  Occasion: {', '.join(o.value for o in item.occasions)}\n\n"
        
        return response
    
    def _handle_style_advice(self, message: str) -> str:
        """Handle style advice requests"""
        message_lower = message.lower()
        
        # Extract color/item to give advice on
        if "white" in message_lower:
            return ("🧥 Style Tip: White pieces are versatile basics!\n\n"
                   "✅ Pairs well with: Black, navy, beige, denim, pastels\n"
                   "❌ Avoid: White-on-white (can look incomplete)\n\n"
                   "💡 Style ideas:\n"
                   "• White tee + jeans + sneakers = classic casual\n"
                   "• White shirt + dark trousers + loafers = smart casual\n"
                   "• White layers under jackets for depth")
        
        elif "black" in message_lower:
            return ("🧥 Style Tip: Black is timeless and slimming!\n\n"
                   "✅ Pairs well with: White, gray, navy, bold colors\n"
                   "❌ Avoid: All black (can feel heavy)\n\n"
                   "💡 Style ideas:\n"
                   "• Black jacket + white tee + blue jeans = effortless cool\n"
                   "• Black pants + colorful top = balanced statement")
        
        elif "jeans" in message_lower or "denim" in message_lower:
            return ("👖 Style Tip: Denim is the ultimate wardrobe staple!\n\n"
                   "✅ Pairs well with: White tees, blazers, sneakers\n"
                   "❌ Avoid: Too much denim (avoid denim-on-denim unless styled)\n\n"
                   "💡 Style ideas:\n"
                   "• Blue jeans + white tee + sneakers = easy casual\n"
                   "• Black jeans + fitted top + boots = night out")
        
        else:
            return ("💡 Style Advice:\n\n"
                   "To get personalized style advice, tell me:\n"
                   "• What item you're styling\n"
                   "• What occasion it's for\n"
                   "• Your color preferences\n\n"
                   "Example: 'How do I style a navy blazer for a date?'")
    
    def _handle_feedback(self, message: str) -> str:
        """Handle user feedback on recommendations"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["like", "love", "great", "perfect", "awesome"]):
            return ("I'm so glad you like it! 😊\n\n"
                   "Would you like me to:\n"
                   "• Generate more outfits in a similar style?\n"
                   "• Save this as a favorite combination?\n"
                   "• Find similar items?")
        
        elif any(word in message_lower for word in ["not", "don't", "dislike", "boring", "plain"]):
            return ("Got it! Let me adjust my recommendations.\n\n"
                   "Tell me more about what you're looking for:\n"
                   "• More colorful?\n"
                   "• More casual?\n"
                   "• More formal?\n"
                   "• Different style altogether?\n\n"
                   "I want to find the perfect outfit for you!")
        
        return "Thanks for your feedback! Let me know if you'd like more recommendations."
    
    def _handle_help(self, message: str) -> str:
        """Show help information"""
        return ("👗 **Fashion Recommender Capabilities**\n\n"
               "**What I can do:**\n"
               "• 🎨 Generate complete outfits for any occasion\n"
               "• 🔍 Search for specific clothing items\n"
               "• 💡 Give style advice and tips\n"
               "• 👔 Match pieces together\n"
               "• 📅 Recommend outfits for different seasons\n\n"
               "**How to use:**\n"
               "• 'Suggest an outfit for a date tonight'\n"
               "• 'What should I wear to work tomorrow?'\n"
               "• 'Generate a casual spring outfit'\n"
               "• 'Find blue jeans'\n"
               "• 'How do I style a white shirt?'\n\n"
               "Just chat with me naturally!")
    
    def _handle_greeting(self, message: str) -> str:
        """Handle greeting"""
        return ("👋 Hello! I'm your personal fashion stylist!\n\n"
               "I can help you:\n"
               "• 👗 Put together complete outfits\n"
               "• 🎯 Match pieces you already have\n"
               "• 💄 Give style advice\n"
               "• 🔥 Keep you on-trend\n\n"
               "What are you looking to wear today?")
    
    def _handle_general(self, message: str) -> str:
        """Handle general messages"""
        return ("I didn't quite catch that! 😅\n\n"
               "Try asking me:\n"
               "• 'Recommend an outfit for a date'\n"
               "• 'What should I wear to work?'\n"
               "• 'Generate a casual spring look'\n"
               "• 'Find sneakers'\n"
               "• 'How do I style black jeans?'\n\n"
               "Or ask for help to see what I can do!")
    
    def _format_outfit_response(self, outfit: Dict) -> str:
        """Format outfit for display"""
        if "error" in outfit:
            return "I couldn't find a matching outfit. Try adjusting your preferences!"
        
        response = f"✨ {self._generate_outfit_title(outfit)}\n\n"
        response += self._format_outfit_items(outfit)
        response += self._format_recommendations(outfit)
        
        return response
    
    def _generate_outfit_title(self, outfit: Dict) -> str:
        """Generate a catchy title for the outfit"""
        season = outfit.get("season", "").title()
        occasion = outfit.get("occasion", "").title()
        style = outfit.get("style", "").title()
        
        templates = [
            f"Perfect {season} {occasion} Look ✨",
            f"{season} Ready: {style} Vibes 🎯",
            f"{occasion} Goals: {style} Edit 🌟",
            f"Your {season} {style} Ensemble 💫",
        ]
        
        import random
        return random.choice(templates)
    
    def _format_outfit_items(self, outfit: Dict) -> str:
        """Format outfit items for display"""
        response = "**整套穿搭 (Complete Outfit):**\n\n"
        
        layers = outfit.get("layers", {})
        
        if layers.get("base_layer"):
            item = layers["base_layer"]
            response += f"👕 上衣: {item['name']}\n"
            response += f"   Color: {item['color']}\n"
            if item.get('image_url'):
                response += f"   Image: {item['image_url']}\n"
            response += "\n"
        
        if layers.get("outerwear"):
            item = layers["outerwear"]
            response += f"🧥 外套: {item['name']}\n"
            response += f"   Color: {item['color']}\n"
            if item.get('image_url'):
                response += f"   Image: {item['image_url']}\n"
            response += "\n"
        
        if layers.get("bottom"):
            item = layers["bottom"]
            response += f"👖 下装: {item['name']}\n"
            response += f"   Color: {item['color']}\n"
            if item.get('image_url'):
                response += f"   Image: {item['image_url']}\n"
            response += "\n"
        
        if layers.get("footwear"):
            item = layers["footwear"]
            response += f"👟 鞋子: {item['name']}\n"
            response += f"   Color: {item['color']}\n"
            if item.get('image_url'):
                response += f"   Image: {item['image_url']}\n"
            response += "\n"
        
        if layers.get("accessories"):
            response += "配饰 Accessories:\n"
            for acc in layers["accessories"]:
                response += f"  • {acc['name']} ({acc['color']})"
                if acc.get('image_url'):
                    response += f" - {acc['image_url']}"
                response += "\n"
            response += "\n"
        
        # Color palette
        response += f"🎨 Color Palette: {' + '.join(outfit.get('color_palette', []))}\n"
        
        return response
    
    def _format_recommendations(self, outfit: Dict) -> str:
        """Format recommendations"""
        recs = outfit.get("recommendations", [])
        response = ""
        
        if recs:
            response += "\n💡 **Style Notes:**\n"
            for rec in recs:
                response += f"• {rec}\n"
        
        return response
    
    def generate_outfit_image_prompt(self, outfit: Dict) -> str:
        """Generate image generation prompt for the outfit"""
        items = []
        
        layers = outfit.get("layers", {})
        
        if layers.get("base_layer"):
            items.append(layers["base_layer"]["name"])
        if layers.get("outerwear"):
            items.append(layers["outerwear"]["name"])
        if layers.get("bottom"):
            items.append(layers["bottom"]["name"])
        if layers.get("footwear"):
            items.append(layers["footwear"]["name"])
        
        colors = outfit.get("color_palette", [])
        
        prompt = (
            f"Fashion outfit flat lay photography, "
            f"{', '.join(items)}, "
            f"color palette: {', '.join(colors)}, "
            f"minimalist studio background, "
            f"professional e-commerce photography, "
            f"clean aesthetic, "
            f"high quality, 4K"
        )
        
        return prompt


def demo():
    """Demo the fashion chatbot"""
    print("\n" + "=" * 60)
    print("👗 Fashion Recommender Chatbot Demo")
    print("=" * 60)
    
    # Initialize chatbot
    chatbot = FashionChatbot()
    
    # Demo conversations
    conversations = [
        "What should I wear for a date tonight?",
        "Generate a casual spring outfit",
        "Find blue jeans",
        "How do I style a white shirt?",
    ]
    
    for msg in conversations:
        print(f"\n👤 User: {msg}")
        print("-" * 40)
        response = chatbot.chat(msg)
        print(f"🤖 Bot: {response}")
        print()


if __name__ == "__main__":
    demo()

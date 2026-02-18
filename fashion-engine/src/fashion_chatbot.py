"""
Fashion Recommender Chatbot
===========================
LLM-powered fashion recommendation with free LLM models.
Supports: MiniMax, Groq, DeepSeek, Hugging Face, Google Gemini
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

from rules.outfit_assembler import (
    OutfitAssembler, Season, Occasion, Style, 
    FashionItem, SAMPLE_DATABASE
)
from src.llm_factory import LLMFactory


class FashionChatbot:
    """
    Fashion chatbot with LLM capabilities.
    Uses free tier models: MiniMax → Groq → DeepSeek → HuggingFace → Gemini
    """
    
    def __init__(self):
        self.assembler = OutfitAssembler()
        self.llm = LLMFactory()  # Auto-configures with best available free model
        self.conversation_history: List[Dict] = []
        
        # Initialize with sample database
        for item in SAMPLE_DATABASE:
            self.assembler.add_item(item)
    
    def chat(self, message: str) -> str:
        """
        Main chat interface.
        Uses LLM for natural language understanding when available,
        falls back to rule-based logic otherwise.
        """
        self.conversation_history.append({"role": "user", "content": message})
        
        # Detect intent
        intent = self._detect_intent(message)
        
        # Generate response
        response = self._handle_intent(intent, message)
        
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _detect_intent(self, message: str) -> str:
        """Detect user intent"""
        message_lower = message.lower()
        
        if any(w in message_lower for w in ["recommend", "suggest", "outfit", "what to wear"]):
            return "recommendation"
        elif any(w in message_lower for w in ["find", "search", "look for"]):
            return "search"
        elif any(w in message_lower for w in ["put together", "assemble", "create outfit"]):
            return "outfit_generation"
        elif any(w in message_lower for w in ["style advice", "how to style", "what goes with"]):
            return "style_advice"
        elif any(w in message_lower for w in ["like", "love", "great", "not my style"]):
            return "feedback"
        elif any(w in message_lower for w in ["help", "what can you do"]):
            return "help"
        elif any(w in message_lower for w in ["hello", "hi", "hey"]):
            return "greeting"
        
        return "general"
    
    def _handle_intent(self, intent: str, message: str) -> str:
        """Route to appropriate handler"""
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
        """Extract season, occasion, style from message"""
        context = {"season": None, "occasion": None, "style": None}
        msg_lower = message.lower()
        
        # Seasons
        season_map = {"spring": Season.SPRING, "summer": Season.SUMMER, 
                     "fall": Season.FALL, "winter": Season.WINTER}
        for word, season in season_map.items():
            if word in msg_lower:
                context["season"] = season
                break
        
        # Occasions
        occasion_map = {"casual": Occasion.CASUAL, "work": Occasion.WORK,
                       "date": Occasion.DATE, "party": Occasion.PARTY,
                       "formal": Occasion.FORMAL, "sport": Occasion.SPORT}
        for word, occasion in occasion_map.items():
            if word in msg_lower:
                context["occasion"] = occasion
                break
        
        # Styles
        style_map = {"casual": Style.CASUAL, "formal": Style.FORMAL,
                    "trendy": Style.TRENDY, "athletic": Style.ATHLETIC,
                    "minimalist": Style.MINIMALIST, "vintage": Style.VINTAGE}
        for word, style in style_map.items():
            if word in msg_lower:
                context["style"] = style
                break
        
        return context
    
    def _handle_recommendation(self, message: str) -> str:
        """Generate outfit recommendation"""
        context = self._parse_context(message)
        
        season = context.get("season") or Season.ALL_SEASON
        occasion = context.get("occasion") or Occasion.CASUAL
        style = context.get("style")
        
        outfit = self.assembler.generate_outfit_set(
            season=season,
            occasion=occasion,
            style_preference=style
        )
        
        return self._format_outfit(outfit)
    
    def _handle_outfit_generation(self, message: str) -> str:
        """Generate complete outfit set"""
        context = self._parse_context(message)
        
        season = context.get("season") or Season.SPRING
        occasion = context.get("occasion") or Occasion.CASUAL
        style = context.get("style")
        
        outfit = self.assembler.generate_outfit_set(
            season=season,
            occasion=occasion,
            style_preference=style,
            include_accessories=True
        )
        
        response = self._format_outfit(outfit)
        
        # Add image prompt
        prompt = self._generate_image_prompt(outfit)
        response += f"\n\n🖼️ Image Prompt:\n{prompt}"
        
        return response
    
    def _handle_search(self, message: str) -> str:
        """Search for items"""
        msg_lower = message.lower()
        results = []
        
        for item_id, item in self.assembler.items.items():
            if (msg_lower in item.name.lower() or 
                msg_lower in item.category.lower() or
                msg_lower in item.subcategory.lower()):
                results.append(item)
        
        if not results:
            return "No items found matching your search."
        
        response = f"🔍 Found {len(results)} items:\n\n"
        for item in results[:5]:
            response += f"• {item.name} ({item.category})\n"
            response += f"  Color: {item.color.value} | Style: {item.styles[0].value}\n\n"
        
        return response
    
    def _handle_style_advice(self, message: str) -> str:
        """Give style advice using LLM"""
        msg_lower = message.lower()
        
        # Use LLM for personalized advice if available
        if self.llm.get_default():
            prompt = f"""
You are a fashion stylist. Give advice for: {message}

Consider:
- Color coordination
- Occasion appropriateness  
- Current trends
- Versatility

Keep response concise and helpful.
"""
            return self.llm.generate(prompt)
        
        # Fallback to rule-based
        if "white" in msg_lower:
            return ("🧥 Style Tip: White pieces are versatile!\n\n"
                   "✅ Pairs with: Black, navy, beige, denim\n"
                   "❌ Avoid: White-on-white\n\n"
                   "💡 Ideas: White tee + jeans + sneakers")
        
        return "Tell me what item you want styling advice for!"
    
    def _handle_feedback(self, message: str) -> str:
        """Handle user feedback"""
        if "not" in message.lower() or "don't" in message.lower():
            return ("Got it! Let me adjust.\n\n"
                   "Tell me what you prefer:\n"
                   "• More colorful?\n"
                   "• More casual?\n"
                   "• Different style?")
        
        return "Great to hear! Want more outfits like this?"
    
    def _handle_help(self, message: str) -> str:
        """Show help"""
        return ("👗 **Fashion Bot Capabilities:**\n\n"
               "• 🎨 Generate complete outfits\n"
               "• 🔍 Search for clothing items\n"
               "• 💡 Style advice\n"
               "• 📅 Season-specific recommendations\n\n"
               "Try: 'Suggest a date outfit' or 'Find jeans'")
    
    def _handle_greeting(self, message: str) -> str:
        """Greeting"""
        return ("👋 Hello! Your fashion stylist here!\n\n"
               "I can help with:\n"
               "• 👗 Complete outfit recommendations\n"
               "• 🎯 Personalized style advice\n"
               "• 🔥 Trend updates\n\n"
               "What would you like to wear today?")
    
    def _handle_general(self, message: str) -> str:
        """General response"""
        return ("I didn't catch that! 😅\n\n"
               "Try:\n"
               "• 'Suggest an outfit for a date'\n"
               "• 'What to wear to work?'\n"
               "• 'Generate a casual spring look'\n"
               "• 'Find sneakers'")
    
    def _format_outfit(self, outfit: Dict) -> str:
        """Format outfit for display"""
        if "error" in outfit:
            return "Couldn't generate outfit. Try adjusting your request!"
        
        import random
        
        titles = [
            f"✨ {outfit['season'].title()} {outfit['occasion'].title()} Look",
            f"💫 {outfit['style'].title()} Vibes",
            f"🌟 Your {outfit['season'].title()} Ensemble",
        ]
        title = random.choice(titles)
        
        response = f"{title}\n{'='*50}\n"
        response += f"📊 Compatibility Score: {outfit['compatibility_score']}\n\n"
        
        layers = outfit.get("layers", {})
        
        mapping = {
            "base_layer": ("👕 上衣", "Top"),
            "outerwear": ("🧥 外套", "Outerwear"),
            "bottom": ("👖 下装", "Bottom"),
            "footwear": ("👟 鞋子", "Footwear"),
        }
        
        for key, (cn, en) in mapping.items():
            if layers.get(key):
                item = layers[key]
                response += f"{cn} {en}: {item['name']}\n"
                response += f"   🎨 {item['color']}\n"
        
        if layers.get("accessories"):
            response += "\n💎 配饰 Accessories:\n"
            for acc in layers["accessories"]:
                response += f"   • {acc['name']} ({acc['color']})\n"
        
        response += f"\n🎨 配色: {' + '.join(outfit['color_palette'])}"
        
        if outfit.get("recommendations"):
            response += f"\n\n💡 Tips:\n"
            for rec in outfit["recommendations"][:2]:
                response += f"   • {rec}\n"
        
        return response
    
    def _generate_image_prompt(self, outfit: Dict) -> str:
        """Generate image generation prompt"""
        items = []
        colors = outfit.get("color_palette", [])
        
        layers = outfit.get("layers", {})
        for key in ["base_layer", "outerwear", "bottom", "footwear"]:
            if layers.get(key):
                items.append(layers[key]["name"])
        
        prompt = (
            f"Fashion flat lay, {', '.join(items)}, "
            f"colors: {', '.join(colors)}, "
            f"minimalist studio background, "
            f"professional e-commerce photography, clean aesthetic"
        )
        
        return prompt


def demo():
    """Demo the chatbot"""
    print("\n" + "🎨"*40)
    print("   Fashion Recommender Chatbot")
    print("   免费 LLM 支持 - MiniMax/Groq/DeepSeek")
    print("🎨"*40 + "\n")
    
    chatbot = FashionChatbot()
    
    # Check LLM status
    print("🤖 LLM Status:")
    available = chatbot.llm.list_available()
    for provider in available:
        print(f"   {provider}")
    
    print("\n" + "-"*50)
    
    # Demo conversations
    tests = [
        "What should I wear for a date tonight?",
        "Generate a casual spring outfit",
        "Find white shirts",
        "How do I style a black jacket?",
    ]
    
    for msg in tests:
        print(f"\n👤 User: {msg}")
        print("-"*40)
        response = chatbot.chat(msg)
        print(f"🤖 Bot: {response[:300]}{'...' if len(response) > 300 else ''}")
    
    print("\n" + "="*50)
    print("Demo complete! ✅")


if __name__ == "__main__":
    demo()

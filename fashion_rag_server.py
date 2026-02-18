"""
Fashion RAG Server - Flask Backend
"""
import json
import openai
import chromadb
from flask import Flask, request, jsonify, render_template
import os
import sys

app = Flask(__name__, template_folder='templates', static_folder='.')

# 配置静态文件目录
@app.route('/test/<path:filename>')
def serve_test_image(filename):
    """提供 test 目录的图片"""
    from flask import send_from_directory
    return send_from_directory('.', f'test/{filename}')

# 配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai.api_key = OPENAI_API_KEY

# 缓存数据
DATA_CACHE = {
    "categories": {},
    "attributes": {},
    "image_id_to_file": {},
    "image_data": []
}

def rgb_to_fashion_color(r, g, b):
    """将RGB转换为时尚颜色名称"""
    brightness = (r + g + b) / 3
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    
    # 白/黑 (放宽阈值)
    if brightness > 180:
        return "white"
    elif brightness < 30:
        return "black"
    
    # 灰色
    if max_val - min_val < 25:
        if brightness > 180:
            return "gray"
        elif brightness < 80:
            return "black"
        else:
            return "gray"
    
    # 蓝色
    if b > r and b > g:
        if r < 80 and g < 80:
            return "blue"
        elif r > 100 and g > 100:
            return "purple"
        else:
            return "blue"
    
    # 红色系
    if r > g + 30 and r > b + 30:
        if g > 100 and b > 100:
            return "brown"
        elif g > 150:
            return "orange"
        elif b > 150:
            return "pink"
        else:
            return "red"
    
    # 绿色
    if g > r + 30 and g > b + 30:
        return "green"
    
    # 黄色
    if r > 180 and g > 150 and b < 100:
        return "yellow"
    
    # 紫色
    if r > 100 and b > 100 and g < r - 30:
        return "purple"
    
    # 棕色
    if r > 80 and g > 60 and b > 40 and max_val - min_val < 80:
        return "brown"
    
    # 默认
    if brightness > 150:
        return "gray"
    else:
        return "black"

def extract_color_from_clothing_regions(file_path, image_id):
    """从图片中衣服区域提取颜色（使用Fashionpedia的bbox标注）"""
    try:
        from PIL import Image
        import json
        
        # 加载 Fashionpedia 标注
        with open("instances_attributes_val2020.json", "r") as f:
            data = json.load(f)
        
        # 找到该图片的标注
        bboxes = []
        for ann in data["annotations"]:
            if ann["image_id"] == image_id:
                bbox = ann.get("bbox", [])
                if len(bbox) == 4:
                    bboxes.append(bbox)
        
        if not bboxes:
            return rgb_to_fashion_color(128, 128, 128)
        
        # 打开图片
        img = Image.open(file_path)
        img = img.convert("RGB")
        
        # 收集衣服区域的像素
        all_pixels = []
        bright_pixels = []
        
        for bbox in bboxes:
            x, y, w, h = [int(v) for v in bbox]
            x, y = max(0, x), max(0, y)
            w = min(w, img.width - x)
            h = min(h, img.height - y)
            
            if w <= 0 or h <= 0:
                continue
            
            region = img.crop((x, y, x + w, y + h))
            pixels = list(region.getdata())
            
            for p in pixels:
                if len(p) >= 3:
                    r, g, b = p[0], p[1], p[2]
                    brightness = (r + g + b) / 3
                    all_pixels.append((r, g, b, brightness))
                    
                    if brightness > 180:
                        bright_pixels.append((r, g, b, brightness))
        
        if not all_pixels:
            return rgb_to_fashion_color(128, 128, 128)
        
        # 如果大部分像素是亮的，可能是白色衣服
        if len(bright_pixels) > len(all_pixels) * 0.7:
            r = sum(p[0] for p in bright_pixels) // len(bright_pixels)
            g = sum(p[1] for p in bright_pixels) // len(bright_pixels)
            b = sum(p[2] for p in bright_pixels) // len(bright_pixels)
            return rgb_to_fashion_color(r, g, b)
        
        # 否则用所有像素的平均
        r = sum(p[0] for p in all_pixels) // len(all_pixels)
        g = sum(p[1] for p in all_pixels) // len(all_pixels)
        b = sum(p[2] for p in all_pixels) // len(all_pixels)
        
        return rgb_to_fashion_color(r, g, b)
    
    except Exception as e:
        return "unknown"


def infer_all_metadata(description, file_path, image_id=None):
    """从图片描述推断所有 metadata"""
    desc = description.lower()
    
    # ===== COLOR =====
    color = "unknown"
    color_keywords = {
        "red": ["red", "burgundy", "wine", "maroon"],
        "blue": ["blue", "navy", "denim", "cobalt", "indigo"],
        "black": ["black"],
        "white": ["white"],
        "gray": ["gray", "grey"],
        "green": ["green", "olive", "mint"],
        "yellow": ["yellow", "gold", "mustard"],
        "pink": ["pink", "rose"],
        "purple": ["purple", "lavender", "violet"],
        "orange": ["orange", "coral", "peach"],
        "brown": ["brown", "tan", "beige", "camel", "chocolate"],
    }
    for color_name, keywords in color_keywords.items():
        if any(kw in desc for kw in keywords):
            color = color_name
            break
    
    # 如果描述中没有找到颜色，从衣服区域提取
    if color == "unknown" and file_path and image_id:
        color = extract_color_from_clothing_regions(file_path, image_id)
    
    # ===== GENDER =====
    gender = "unisex"
    male_keywords = ["suit", "tie", "blazer", "dress pants", "oxford", "loafers", "chelsea boots", "belt", "tie", "trousers", "t-shirt", "tshirt", "sweatshirt", "jacket", "coat", "parka", "scarf", "beanie"]
    female_keywords = ["dress", "skirt", "blouse", "heels", "stiletto", "gown", "halter", "crop top", "mini", "midi", "maxi", "cardigan", "top", "tank", "leggings", "bra", "bikini"]
    
    # 如果同时有男性和女性关键词，默认女性（因为Fashionpedia数据集中女性衣服更多）
    has_male = any(kw in desc for kw in male_keywords)
    has_female = any(kw in desc for kw in female_keywords)
    
    if has_male and not has_female:
        gender = "male"
    elif has_female and not has_male:
        gender = "female"
    elif has_male and has_female:
        # 两者都有，检查更多细节
        gender = "unisex"
    else:
        gender = "unisex"
    
    # ===== SEASON / WARMTH =====
    winter_keywords = ["coat", "puffer", "parka", "fur", "wool", "knit sweater", "turtleneck", "boots", "gloves", "scarf", "tights", "stockings", "leather jacket", "winter"]
    summer_keywords = ["shorts", "tank top", "sandals", "bikini", "swimsuit", "halter", "crop top", "mini skirt", "sun dress", "summer"]
    spring_fall_keywords = ["shirt", "blouse", "jeans", "pants", "jacket", "cardigan", "layer", "midi", "maxi", "dress", "sweater", "long sleeve"]
    
    if any(kw in desc for kw in winter_keywords):
        season = "winter"
        warmth = 5
    elif any(kw in desc for kw in summer_keywords):
        season = "summer"
        warmth = 1
    elif any(kw in desc for kw in spring_fall_keywords):
        if "long sleeve" in desc or "sweater" in desc or "knit" in desc:
            season = "fall"
            warmth = 4
        else:
            season = "spring"
            warmth = 3
    else:
        season = "all"
        warmth = 3
    
    # ===== OCCASION / FORMALITY =====
    formal_keywords = ["suit", "gown", "tuxedo", "dress pants", "blazer", "formal"]
    casual_keywords = ["jeans", "shorts", "t-shirt", "sneakers", "casual", "sweatpants"]
    
    if any(kw in desc for kw in formal_keywords):
        formality = "formal"
        occasion = "formal"
    elif any(kw in desc for kw in casual_keywords):
        formality = "casual"
        occasion = "casual"
    else:
        formality = "any"
        occasion = "any"
    
    # ===== STYLE =====
    style = "modern"  # default
    
    # ===== FABRIC =====
    fabric = "any"
    fabric_keywords = {
        "cotton": ["cotton"],
        "silk": ["silk"],
        "linen": ["linen"],
        "denim": ["denim", "jeans"],
        "wool": ["wool", "knit", "sweater"]
    }
    for fabric_name, keywords in fabric_keywords.items():
        if any(kw in desc for kw in keywords):
            fabric = fabric_name
            break
    
    return {
        "season": season,
        "warmth": warmth,
        "color": color,
        "gender": gender,
        "formality": formality,
        "occasion": occasion,
        "style": style,
        "fabric": fabric
    }

def load_fashion_data():
    """加载 Fashionpedia 数据"""
    if DATA_CACHE["image_data"]:
        return DATA_CACHE
    
    print("Loading Fashionpedia data...", flush=True)
    
    # Load corrected data with color + style
    if os.path.exists("corrected_data.json"):
        print("Using corrected data with style & color!", flush=True)
        with open("corrected_data.json", "r") as f:
            corrected = json.load(f)
            DATA_CACHE["image_data"] = corrected["image_data"]
            return DATA_CACHE
    
    with open("instances_attributes_val2020.json", "r") as f:
        data = json.load(f)
    
    DATA_CACHE["categories"] = {cat["id"]: cat["name"] for cat in data["categories"]}
    DATA_CACHE["attributes"] = {attr["id"]: attr["name"] for attr in data["attributes"]}
    DATA_CACHE["image_id_to_file"] = {img["id"]: img["file_name"] for img in data["images"]}
    
# 生成图片描述
    for img in data["images"][:1158]:
        annotations = [ann for ann in data["annotations"] if ann["image_id"] == img["id"]]
        if annotations:
            desc_parts = []
            for ann in annotations[:3]:
                cat = DATA_CACHE["categories"].get(ann["category_id"], "unknown")
                attrs = [DATA_CACHE["attributes"].get(aid, "") for aid in ann.get("attribute_ids", [])]
                desc_parts.append(f"{cat}: {', '.join(attrs)}")
            description = f"Contains: {'; '.join(desc_parts)}"
            
            # 推断所有 metadata
            file_path = f"test/{DATA_CACHE['image_id_to_file'].get(img['id'], '')}"
            image_id = img['id']
            metadata = infer_all_metadata(description, file_path, image_id)
            
            DATA_CACHE["image_data"].append({
                "description": description,
                "image_id": str(img["id"]),
                "file_path": f"test/{DATA_CACHE['image_id_to_file'].get(img['id'], '')}",
                "season": metadata["season"],
                "warmth_level": metadata["warmth"],
                "color": metadata["color"],
                "gender": metadata["gender"],
                "formality": metadata["formality"],
                "occasion": metadata["occasion"],
                "style": metadata["style"],
                "fabric": metadata["fabric"]
            })
    
    print(f"Loaded {len(DATA_CACHE['image_data'])} images", flush=True)
    return DATA_CACHE

def rag_query(user_profile, n_results=5):
    """RAG 查询"""
    print("\n" + "="*60, flush=True)
    print("🔄 RAG PROCESS STARTED", flush=True)
    print("="*60, flush=True)
    
    data = load_fashion_data()
    print(f"✅ Step 1: Loaded {len(data['image_data'])} images from Fashionpedia", flush=True)
    
    # ===== NEW: LLM 分析用户输入 =====
    print("✅ Step 2: Calling LLM to parse user input...", flush=True)
    user_input = user_profile.get('identity', '') or user_profile.get('occasion', '') or 'casual style'
    
    parse_prompt = f"""
Analyze this fashion request and extract structured information:

User's request: "{user_input}"

Extract the following (return JSON only, no other text):
{{
    "age": "age mentioned or empty",
    "identity": "who they are (student, professional, etc.) or empty", 
    "gender": "male|female",
    "body_type": "slim|athletic|curvy|average|any",
    "skin_tone": "fair|medium|dark|any",
    "age_group": "teen|young|middle-aged|senior|any",
    "occasion": "casual|business|formal|date|party|sport|workout|business-casual",
    "formality": "casual|smart-casual|formal|black-tie|any",
    "season": "spring|summer|fall|winter",
    "weather": "hot|warm|cool|cold|any",
    "location": "indoor|outdoor|beach|city|any",
    "time_of_day": "morning|afternoon|evening|any",
    "style": "casual|formal|elegant|edgy|modern|vintage|streetwear",
    "color": "red|blue|black|white|gray|green|yellow|pink|purple|orange|brown|any",
    "fabric": "cotton|silk|linen|denim|wool|any",
    "brand": "luxury|streetwear|vintage|any",
    "budget": "low|medium|high|any",
    "special_needs": "Extract any special requirements like: eye-catching, comfortable, unique, elegant, trendy, vintage, bold, subtle, etc. If none mentioned, use empty string."
}}

IMPORTANT: Pay special attention to special_needs:
- "eye catching" / "stand out" / "attention" → "eye-catching outfit"
- "comfortable" / "comfy" → "comfortable"
- "unique" / "different" → "unique style"
- "elegant" / "sophisticated" → "elegant"
- "bold" / "dramatic" → "bold look"
- "subtle" / "understated" → "subtle look"
- "trendy" / "fashionable" → "trendy"
- "casual" / "relaxed" → "casual"

IMPORTANT OCCASION MAPPINGS:
- "company party" / "work party" / "office party" → "business-casual"
- "business dinner" / "work dinner" → "business"
- "casual friday" / "casual day" → "casual"
- "networking event" → "business-casual"

If not mentioned, ask user for gender preference: male or female, season=spring, formality=any, weather=any, location=any, time_of_day=any, style=modern, color=any, fabric=any, brand=any, budget=any, special_needs=""
"""
    
    parsed_data = {}  # Initialize for debug output
    try:
        parse_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract fashion preferences from user input. Return only valid JSON."},
                {"role": "user", "content": parse_prompt}
            ],
            temperature=0.3
        )
        
        import json
        parsed = parse_response.choices[0].message.content
        # Extract JSON from response
        try:
            parsed_data = json.loads(parsed)
        except:
            # Try to extract JSON from text
            import re
            match = re.search(r'\{[^}]+\}', parsed)
            if match:
                parsed_data = json.loads(match.group())
            else:
                parsed_data = {}
        
        print(f"   ✅ Parsed: {parsed_data}", flush=True)
        
        # Update user_profile with parsed data
        if parsed_data.get('occasion'):
            user_profile['occasion'] = parsed_data['occasion']
        if parsed_data.get('season'):
            user_profile['season'] = parsed_data.get('season', 'spring')
        if parsed_data.get('weather'):
            user_profile['weather'] = parsed_data['weather']
        if parsed_data.get('color'):
            user_profile['color'] = parsed_data['color']
        if parsed_data.get('style'):
            user_profile['style'] = parsed_data['style']
        if parsed_data.get('formality'):
            user_profile['formality'] = parsed_data['formality']
        if parsed_data.get('gender'):
            user_profile['gender'] = parsed_data['gender']
        if parsed_data.get('body_type'):
            user_profile['body_type'] = parsed_data['body_type']
        if parsed_data.get('skin_tone'):
            user_profile['skin_tone'] = parsed_data['skin_tone']
        if parsed_data.get('age_group'):
            user_profile['age_group'] = parsed_data['age_group']
        if parsed_data.get('location'):
            user_profile['location'] = parsed_data['location']
        if parsed_data.get('time_of_day'):
            user_profile['time_of_day'] = parsed_data['time_of_day']
        if parsed_data.get('fabric'):
            user_profile['fabric'] = parsed_data['fabric']
        if parsed_data.get('brand'):
            user_profile['brand'] = parsed_data['brand']
        if parsed_data.get('budget'):
            user_profile['budget'] = parsed_data['budget']
        if parsed_data.get('special_needs'):
            user_profile['special_needs'] = parsed_data['special_needs']
            
    except Exception as e:
        print(f"   ⚠️ LLM parse failed: {e}", flush=True)
    
    # 构建查询
    weather = user_profile.get('weather', '')
    color = user_profile.get('color', '')
    formality = user_profile.get('formality', '')
    gender = user_profile.get('gender', '')
    body_type = user_profile.get('body_type', '')
    location = user_profile.get('location', '')
    time_of_day = user_profile.get('time_of_day', '')
    fabric = user_profile.get('fabric', '')
    budget = user_profile.get('budget', '')
    
    query = f"User: {user_profile['identity']}, {user_profile['gender']} clothing, {user_profile['style']} style, {user_profile['age']}, body={body_type} - Occasion: {user_profile['occasion']}, Formality: {formality}, Season: {user_profile['season']}, Weather: {weather}, Location: {location}, Time: {time_of_day} - Color: {color}, Fabric: {fabric}, Budget: {budget} - Needs: {user_profile['special_needs']}"
    print(f"✅ Step 3: Built query: {query}", flush=True)
    
    # 初始化向量数据库
    print("✅ Step 4: Initializing ChromaDB vector database...", flush=True)
    from chromadb.utils import embedding_functions
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
    )
    
    # 使用持久化存储，避免 collection 丢失
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 删除已存在的 collection
    try:
        client.delete_collection("fashion_rag_live")
        print("   Deleted existing collection", flush=True)
    except:
        pass
    
    collection = client.create_collection(name="fashion_rag_live", embedding_function=openai_ef)
    print("✅ Step 5: ChromaDB initialized", flush=True)
    
    print("✅ Step 6: Adding documents to vector store...", flush=True)
    documents = [item["description"] for item in data["image_data"]]
    ids = [item["image_id"] for item in data["image_data"]]
    collection.add(documents=documents, ids=ids)
    print(f"   Added {len(documents)} documents to vector store", flush=True)
    
    print("✅ Step 7: Running semantic search...", flush=True)
    results = collection.query(query_texts=[query], n_results=50)  # Get more results for grading
    retrieved_docs = results["documents"][0]
    retrieved_ids = results["ids"][0]
    
    # ===== 分级匹配策略 =====
    # 获取用户需求
    user_color = user_profile.get('color', '').lower()
    user_season = user_profile.get('season', '').lower()
    user_occasion = user_profile.get('occasion', '').lower()
    user_formality = user_profile.get('formality', '').lower()
    user_fabric = user_profile.get('fabric', '').lower()
    user_gender = user_profile.get('gender', 'female').lower()
    
    # 季节相邻映射
    season_adjacent = {
        "winter": ["fall"],
        "fall": ["winter", "spring"],
        "spring": ["fall", "summer"],
        "summer": ["spring"]
    }
    
    def get_match_score(img_id, doc, user_gender="female"):
        """计算图片与用户需求的匹配分数 (0-100)，返回分数和不匹配原因"""
        # 首先检查性别 - 如果不匹配直接返回0分
        gender_match = False
        for item in data["image_data"]:
            if item["image_id"] == img_id:
                img_gender = item.get("gender", "").lower()
                if user_gender == "male":
                    if img_gender in ["male", "unisex"]:
                        gender_match = True
                elif user_gender == "female":
                    if img_gender in ["female", "unisex"]:
                        gender_match = True
                else:  # default to female
                    if img_gender in ["female", "unisex"]:
                        gender_match = True
                break
        
        if not gender_match:
            return 0, [f"Gender mismatch"]
        
        score = 100
        reasons = []
        
        for item in data["image_data"]:
            if item["image_id"] == img_id:
                # 颜色匹配 (权重 30)
                if user_color and user_color not in ['any', 'empty', '']:
                    item_color = item.get("color", "")
                    if item_color == user_color:
                        pass  # 完美匹配
                    elif item_color != "":
                        score -= 15
                        reasons.append(f"Color: {item_color} vs {user_color}")
                    else:
                        score -= 5
                        reasons.append(f"Color: unknown vs {user_color}")
                
                # 季节匹配 (权重 25)
                if user_season and user_season != 'any':
                    item_season = item.get("season", "all")
                    if item_season == user_season:
                        pass
                    elif item_season in season_adjacent.get(user_season, []):
                        score -= 10
                        reasons.append(f"Season: {item_season} vs {user_season}")
                    elif item_season != "all":
                        score -= 15
                        reasons.append(f"Season: {item_season} vs {user_season}")
                
                # 场合匹配 (权重 20)
                if user_occasion and user_occasion != 'any':
                    item_occasion = item.get("occasion", "any")
                    if item_occasion == user_occasion:
                        pass
                    elif item_occasion != "any":
                        score -= 15
                        reasons.append(f"Occasion: {item_occasion} vs {user_occasion}")
                
                # 正式程度匹配 (权重 15)
                if user_formality and user_formality != 'any':
                    item_formality = item.get("formality", "any")
                    if item_formality == user_formality:
                        pass
                    elif item_formality != "any":
                        score -= 10
                        reasons.append(f"Formality: {item_formality} vs {user_formality}")
                
                # 面料匹配 (权重 10)
                if user_fabric and user_fabric != 'any':
                    item_fabric = item.get("fabric", "any")
                    if item_fabric == user_fabric:
                        pass
                    elif item_fabric != "any":
                        score -= 5
                        reasons.append(f"Fabric: {item_fabric} vs {user_fabric}")
                
                break
        
        return max(0, score), reasons
    
    # 计算所有结果的匹配分数
    scored_results = []
    for doc, img_id in zip(retrieved_docs, retrieved_ids):
        score, reasons = get_match_score(img_id, doc, user_gender)
        scored_results.append((score, doc, img_id, reasons))
    
    # 按分数排序（降序）
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    # 如果结果少于10个，放宽条件逐步添加
    final_results = scored_results[:10]
    
    if len(final_results) < 3:
        print(f"   ⚠️ Few results, relaxing filters...", flush=True)
        # 放宽颜色限制
        if user_color and user_color not in ['any', 'empty']:
            for score, doc, img_id, reasons in scored_results[10:30]:
                if img_id not in [r[2] for r in final_results]:
                    # 检查颜色是否接近
                    matched = False
                    for item in data["image_data"]:
                        if item["image_id"] == img_id:
                            if item.get("color") in ["unknown", ""]:
                                matched = True
                            break
                    if matched:
                        final_results.append((score, doc, img_id, reasons))
                        if len(final_results) >= 10:
                            break
    
    retrieved_docs = [r[1] for r in final_results]
    retrieved_ids = [r[2] for r in final_results]
    
    # 打印结果
    print(f"   Retrieved {len(retrieved_ids)} images (scored)", flush=True)
    for i, (score, doc, img_id, reasons) in enumerate(final_results):
        match_level = "✅" if score > 80 else ("🟡" if score > 50 else "🟢")
        print(f"   {match_level} [{i+1}] Image ID: {img_id} (score: {score})", flush=True)
        print(f"       {doc[:60]}...", flush=True)
    
    print("✅ Step 8: Calling GPT-3.5 for styling advice...", flush=True)
    
    # GPT-3.5 生成建议
    context = "\n".join([f"[Image{i+1}] {doc}" for i, doc in enumerate(retrieved_docs)])
    prompt = f"""
You are a professional fashion stylist. Based on the user profile and reference images, give styling advice:

=== User Profile ===
Age: {user_profile['age']}
Identity: {user_profile['identity']}
Family: {user_profile['family']}
Occasion: {user_profile['occasion']}
Theme: {user_profile['theme']}
Style: {user_profile['style']}
Special Needs: {user_profile['special_needs']}

=== Reference Images ===
{context}

Please provide:
1. Overall outfit recommendation (top, bottom, shoes, accessories, colors)
2. Key points for each item
3. Styling tips
4. Color suggestions

Reply in English, be practical and specific.
"""
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional fashion stylist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    advice = response.choices[0].message.content
    print("   ✅ GPT-3.5 response received!", flush=True)
    
    # 获取图片路径和匹配信息
    print("✅ Step 9: Fetching image paths...", flush=True)
    matched_images = []
    for i, img_id in enumerate(retrieved_ids):
        for item in data["image_data"]:
            if item["image_id"] == img_id:
                # 获取该图片的匹配分数和原因
                score = final_results[i][0] if i < len(final_results) else 50
                reasons = final_results[i][3] if i < len(final_results) else []
                
                # 标记匹配级别
                if score >= 95:
                    match_level = "perfect"
                    match_label = "✅ Perfect Match"
                    match_reasons = ""
                elif score >= 70:
                    match_level = "partial"
                    match_label = "🟡 Partial Match"
                    match_reasons = " | ".join(reasons) if reasons else "Some criteria not matched"
                elif score >= 50:
                    match_level = "relaxed"
                    match_label = "🟢 Loose Match"
                    match_reasons = " | ".join(reasons) if reasons else "Limited match"
                else:
                    match_level = "poor"
                    match_label = "🔴 Poor Match"
                    match_reasons = " | ".join(reasons) if reasons else "Low relevance"
                
                item_copy = item.copy()
                item_copy["match_score"] = score
                item_copy["match_level"] = match_level
                item_copy["match_label"] = match_label
                item_copy["match_reasons"] = match_reasons
                matched_images.append(item_copy)
                break
    
    print("="*60, flush=True)
    print("✅ RAG PROCESS COMPLETE!", flush=True)
    print("="*60 + "\n", flush=True)
    
    # Include debug info in response
    return {
        "advice": advice,
        "images": matched_images[:n_results],
        "query": query,
        "debug": {
            "parsed_by_llm": parsed_data if 'parsed_data' in dir() else {},
            "chroma_documents": len(documents),
            "retrieved_count": len(retrieved_ids)
        }
    }

@app.route('/')
def index():
    """渲染网页"""
    return render_template('chat.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """API: 获取推荐"""
    data = request.json
    user_profile = {
        "age": data.get('age', ''),
        "identity": data.get('identity', ''),
        "family": data.get('family', ''),
        "occasion": data.get('occasion', ''),
        "theme": data.get('theme', ''),
        "style": data.get('style', ''),
        "season": data.get('season', ''),
        "weather": data.get('weather', ''),
        "color": data.get('color', ''),
        "formality": data.get('formality', ''),
        "gender": data.get('gender', ''),
        "body_type": data.get('body_type', ''),
        "skin_tone": data.get('skin_tone', ''),
        "age_group": data.get('age_group', ''),
        "location": data.get('location', ''),
        "time_of_day": data.get('time_of_day', ''),
        "fabric": data.get('fabric', ''),
        "brand": data.get('brand', ''),
        "budget": data.get('budget', ''),
        "special_needs": data.get('special_needs', '')
    }
    
    try:
        result = rag_query(user_profile)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/color-filter')
def color_filter():
    """渲染颜色筛选工具"""
    return render_template('color_filter.html')

@app.route('/api/images')
def api_images():
    """API: 获取所有图片及其颜色"""
    data = load_fashion_data()
    images = []
    for item in data["image_data"][:500]:  # Return first 500
        images.append({
            "image_id": item["image_id"],
            "file_path": item["file_path"],
            "color": item.get("color", "unknown")
        })
    return jsonify({"success": True, "images": images})

@app.route('/api/search')
def api_search():
    """API: 搜索图片"""
    query = request.args.get('q', '').lower()
    data = load_fashion_data()
    images = []
    
    for item in data["image_data"]:
        desc = item.get("description", "").lower()
        if query in desc:
            images.append({
                "image_id": item["image_id"],
                "file_path": item["file_path"],
                "color": item.get("color", "unknown"),
                "description": item.get("description", "")[:100]
            })
    
    return jsonify({"success": True, "images": images[:100]})


@app.route('/api/save-corrections', methods=['POST'])
def save_corrections():
    """保存颜色修正"""
    try:
        data = request.json
        corrections = data.get('corrections', {})
        
        # Load existing corrections
        corrections_file = 'color_corrections.json'
        existing = {}
        if os.path.exists(corrections_file):
            with open(corrections_file, 'r') as f:
                existing = json.load(f)
        
        # Merge corrections
        existing.update(corrections)
        
        # Save to file
        with open(corrections_file, 'w') as f:
            json.dump(existing, f, indent=2)
        
        return jsonify({"success": True, "count": len(existing)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/apply-corrections')
def apply_corrections():
    """应用修正到 metadata 并重新生成 ChromaDB"""
    try:
        corrections_file = 'color_corrections.json'
        if not os.path.exists(corrections_file):
            return jsonify({"success": False, "error": "No corrections file"})
        
        with open(corrections_file, 'r') as f:
            corrections = json.load(f)
        
        # Apply corrections to image_data
        count = 0
        for item in DATA_CACHE["image_data"]:
            if item["image_id"] in corrections:
                item["color"] = corrections[item["image_id"]]
                count += 1
        
        # Save corrected data to file (for persistence across restarts)
        with open('corrected_data.json', 'w') as f:
            json.dump({"image_data": DATA_CACHE["image_data"]}, f, indent=2)
        
        # Delete old ChromaDB
        import shutil
        if os.path.exists('chroma_db'):
            shutil.rmtree('chroma_db')
        
        return jsonify({"success": True, "corrected": count})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    print("🚀 Fashion RAG Server started at http://localhost:5001", flush=True)
    app.run(debug=True, port=5001, use_reloader=False)

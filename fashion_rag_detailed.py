"""
Fashion RAG Server - 按品类展示推荐
"""
import json
import openai
import chromadb
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, template_folder='templates', static_folder='.')
CORS(app)  # 允许跨域请求

# 配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai.api_key = OPENAI_API_KEY

# 缓存数据
DATA_CACHE = {"loaded": False}

def load_fashion_data():
    """加载 Fashionpedia 数据"""
    if DATA_CACHE.get("loaded"):
        return DATA_CACHE
    
    print("Loading Fashionpedia data...")
    with open("instances_attributes_val2020.json", "r") as f:
        data = json.load(f)
    
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    attributes = {attr["id"]: attr["name"] for attr in data["attributes"]}
    image_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    
    # 按品类组织图片
    category_images = {}
    for img in data["images"][:1000]:
        annotations = [ann for ann in data["annotations"] if ann["image_id"] == img["id"]]
        if annotations:
            for ann in annotations[:2]:
                cat_id = ann["category_id"]
                cat_name = categories.get(cat_id, "unknown")
                
                if cat_name not in category_images:
                    category_images[cat_name] = []
                
                desc_parts = []
                for ann2 in annotations[:2]:
                    c = categories.get(ann2["category_id"], "unknown")
                    attrs = [attributes.get(aid, "") for aid in ann2.get("attribute_ids", [])]
                    desc_parts.append(f"{c}: {', '.join(attrs)}")
                
                description = f"{'; '.join(desc_parts)}"
                category_images[cat_name].append({
                    "description": description,
                    "image_id": str(img["id"]),
                    "file_path": f"test/{image_id_to_file.get(img['id'], '')}"
                })
    
    DATA_CACHE.update({
        "loaded": True,
        "categories": categories,
        "attributes": attributes,
        "image_id_to_file": image_id_to_file,
        "category_images": category_images
    })
    
    print(f"Loaded {len(category_images)} categories")
    return DATA_CACHE

def analyze_user_needs(user_profile):
    """GPT 分析用户需要什么品类"""
    prompt = f"""
Analyze what clothing categories this person needs based on their profile:

User Profile:
- Age: {user_profile['age']}
- Identity: {user_profile['identity']}
- Occasion: {user_profile['occasion']}
- Theme: {user_profile['theme']}
- Style: {user_profile['style']}
- Special Needs: {user_profile['special_needs']}

Fashionpedia Categories Available:
- dress (连衣裙)
- shirt, blouse (衬衫)
- top, t-shirt, sweatshirt (上衣)
- pants (裤子)
- jeans (牛仔裤)
- skirt (裙子)
- shoes (鞋子)
- bag (包包)
- accessories (配饰)
- outerwear (外套)
- jacket (夹克)
- hat (帽子)

Output format:
{{"reasoning": "brief reasoning", "items": ["item1", "item2", ...], "explanation": "why these items"}}

Example for wedding guest: {{"reasoning": "Needs elegant dress or suit", "items": ["dress", "shoes", "accessories", "outerwear"], "explanation": "For a wedding, an elegant dress is appropriate. Shoes and accessories complete the look."}}
"""
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a fashion stylist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content

def search_by_category(query, category, n_results=2):
    """按品类搜索"""
    data = load_fashion_data()
    
    from chromadb.utils import embedding_functions
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
    )
    
    client = chromadb.Client()
    
    # 删除已存在的集合
    try:
        client.delete_collection(name=f"fashion_{category}")
    except:
        pass
    
    collection = client.create_collection(name=f"fashion_{category}", embedding_function=openai_ef)
    
    # 获取该品类的图片
    cat_images = data["category_images"].get(category, [])[:100]
    if not cat_images:
        return []
    
    documents = [item["description"] for item in cat_images]
    ids = [item["image_id"] for item in cat_images]
    collection.add(documents=documents, ids=ids)
    
    # 搜索
    results = collection.query(query_texts=[query], n_results=n_results)
    
    matched = []
    for doc, img_id in zip(results["documents"][0], results["ids"][0]):
        for item in cat_images:
            if item["image_id"] == img_id:
                matched.append(item)
                break
    
    return matched

@app.route('/test/<path:filename>')
def serve_image(filename):
    return send_from_directory('.', f'test/{filename}')

@app.route('/')
def index():
    return render_template('index_detailed.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_profile = {
        "age": data.get('age', ''),
        "identity": data.get('identity', ''),
        "family": data.get('family', ''),
        "occasion": data.get('occasion', ''),
        "theme": data.get('theme', ''),
        "style": data.get('style', ''),
        "special_needs": data.get('special_needs', '')
    }
    
    try:
        # Step 1: 分析需要什么品类
        print("📊 Analyzing user needs...")
        analysis = analyze_user_needs(user_profile)
        print(f"Analysis result: {analysis[:200]}...")
        
        # 解析分析结果
        import json as json_lib
        try:
            analysis_data = json_lib.loads(analysis)
            needed_items = analysis_data.get("items", ["dress", "shoes", "accessories"])
            explanation = analysis_data.get("explanation", "")
        except:
            # 默认品类
            needed_items = ["dress", "shoes", "accessories"]
            explanation = ""
        
        # Step 2: 按品类搜索图片
        print(f"🔍 Searching for categories: {needed_items}")
        
        query = f"{user_profile['style']} {user_profile['occasion']} {user_profile['special_needs']}"
        
        categorized_results = {}
        for category in needed_items:
            print(f"  📷 Searching {category}...")
            images = search_by_category(query, category, n_results=3)
            categorized_results[category] = images
            print(f"    Found {len(images)} images")
        
        # Step 3: 生成穿搭建议
        print("🤖 Generating styling advice...")
        
        context_parts = []
        for category, images in categorized_results.items():
            if images:
                context_parts.append(f"\n=== {category.upper()} ===")
                for i, img in enumerate(images[:2], 1):
                    context_parts.append(f"{category.title()} {i}: {img['description']}")
        
        context = "\n".join(context_parts)
        
        advice_prompt = f"""
You are a professional fashion stylist. Based on the user profile and reference images, give personalized styling advice.

=== User Profile ===
Age: {user_profile['age']}
Identity: {user_profile['identity']}
Family: {user_profile['family']}
Occasion: {user_profile['occasion']}
Theme: {user_profile['theme']}
Style: {user_profile['style']}
Special Needs: {user_profile['special_needs']}

=== Recommended Items (from RAG) ===
{context}

=== Styling Logic ===
{explanation}

Please provide:
1. **Complete Outfit** - What items to wear together (NO contradiction - if dress is recommended, NO pants)
2. **Item-by-Item Guide** - For each recommended category
3. **Styling Tips** - How to coordinate colors and accessories
4. **Why This Works** - Explain the fashion reasoning

IMPORTANT: If a dress is recommended, do NOT recommend pants/skirts for the same outfit.
IMPORTANT: Consider the user's specific needs (Asian, petite 5ft, wedding guest).

Reply in English, be practical and specific.
"""
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional fashion stylist."},
                {"role": "user", "content": advice_prompt}
            ],
            temperature=0.7
        )
        
        advice = response.choices[0].message.content
        
        return jsonify({
            "success": True,
            "result": {
                "advice": advice,
                "analysis": analysis,
                "categorized_results": categorized_results,
                "user_profile": user_profile
            }
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    print("🚀 Fashion RAG Server started at http://localhost:5006")
    app.run(debug=True, port=5006, use_reloader=False)

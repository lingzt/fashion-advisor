"""
Fashion RAG Server - 支持多数据库（Fashionpedia + DeepFashion2）
"""
import json
import openai
import chromadb
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, template_folder='templates', static_folder='.')
CORS(app)

# 配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai.api_key = OPENAI_API_KEY

# 数据库路径
DB_PATHS = {
    "fashionpedia": "instances_attributes_val2020.json",
    "deepfashion2": "deepfashion2_test.json"  # 需要下载
}

def load_fashionpedia():
    """加载 Fashionpedia 数据"""
    path = DB_PATHS["fashionpedia"]
    if not os.path.exists(path):
        return None
    
    with open(path, "r") as f:
        data = json.load(f)
    
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    attributes = {attr["id"]: attr["name"] for attr in data["attributes"]}
    
    items = []
    for img in data["images"]:
        annotations = [ann for ann in data["annotations"] if ann["image_id"] == img["id"]]
        if annotations:
            desc_parts = []
            for ann in annotations[:2]:
                cat = categories.get(ann["category_id"], "unknown")
                attrs = [attributes.get(aid, "") for aid in ann.get("attribute_ids", [])]
                desc_parts.append(f"{cat}: {', '.join(attrs)}")
            
            items.append({
                "description": f"[Fashionpedia] {'; '.join(desc_parts)}",
                "image_id": f"fp_{img['id']}",
                "file_path": f"test/{img['file_name']}",
                "category": desc_parts[0].split(":")[0] if desc_parts else "unknown",
                "source": "fashionpedia"
            })
    
    return items

def load_deepfashion2():
    """加载 DeepFashion2 数据"""
    path = DB_PATHS["deepfashion2"]
    if not os.path.exists(path):
        return None
    
    with open(path, "r") as f:
        data = json.load(f)
    
    categories = data.get("categories", {})
    
    items = []
    for img in data.get("images", []):
        annotations = img.get("annotations", [])
        if annotations:
            desc_parts = []
            for ann in annotations[:2]:
                cat = categories.get(str(ann.get("category_id", 0)), "unknown")
                attrs = ann.get("attributes", [])
                desc_parts.append(f"{cat}: {', '.join(attrs)}")
            
            items.append({
                "description": f"[DeepFashion2] {'; '.join(desc_parts)}",
                "image_id": f"df2_{img['id']}",
                "file_path": f"deepfashion2_test/{img['file_name']}",
                "category": desc_parts[0].split(":")[0] if desc_parts else "unknown",
                "source": "deepfashion2"
            })
    
    return items

def build_combined_index():
    """构建组合索引"""
    all_items = []
    
    # 加载 Fashionpedia
    fp_items = load_fashionpedia()
    if fp_items:
        all_items.extend(fp_items)
        print(f"✅ Loaded {len(fp_items)} items from Fashionpedia")
    
    # 加载 DeepFashion2
    df2_items = load_deepfashion2()
    if df2_items:
        all_items.extend(df2_items)
        print(f"✅ Loaded {len(df2_items)} items from DeepFashion2")
    
    return all_items

def analyze_and_route(user_profile):
    """GPT 分析需求并决定从哪个数据库检索"""
    prompt = f"""
Analyze what this user needs and which database to use:

User Profile:
- Age: {user_profile['age']}
- Identity: {user_profile['identity']}
- Occasion: {user_profile['occasion']}
- Theme: {user_profile['theme']}
- Style: {user_profile['style']}
- Special Needs: {user_profile['special_needs']}

Databases:
- Fashionpedia: Best for dresses, tops, full outfits, elegant styles
- DeepFashion2: Best for shoes, bags, individual items, casual/trendy styles

Output JSON format:
{{"reasoning": "...", "databases": ["fashionpedia", "deepfashion2"], "categories": ["dress", "shoes"], "explanation": "..."}}

Example:
- Wedding guest, elegant → Fashionpedia + dress
- Daily work, trendy shoes → DeepFashion2 + shoes
- Party outfit → Both + dress, accessories
"""
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a fashion stylist."},
                 {"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content

def rag_query(user_profile, n_results=5):
    """RAG 查询（双数据库）"""
    # 分析需求
    analysis = analyze_and_route(user_profile)
    
    import json as json_lib
    try:
        analysis_data = json_lib.loads(analysis)
    except:
        analysis_data = {"databases": ["fashionpedia"], "categories": ["dress"]}
    
    # 构建查询
    query = f"{user_profile['style']} {user_profile['occasion']} {user_profile['special_needs']}"
    
    # 构建组合索引
    all_items = build_combined_index()
    
    if not all_items:
        return {
            "success": False,
            "error": "No database available. Please download Fashionpedia or DeepFashion2.",
            "analysis": analysis
        }
    
    # 初始化向量数据库
    from chromadb.utils import embedding_functions
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
    )
    
    client = chromadb.Client()
    collection = client.create_collection(name="fashion_combined", embedding_function=openai_ef)
    
    documents = [item["description"] for item in all_items]
    ids = [item["image_id"] for item in all_items]
    collection.add(documents=documents, ids=ids)
    
    # 检索
    results = collection.query(query_texts=[query], n_results=n_results * 2)
    
    # 按数据库分组
    categorized_results = {
        "fashionpedia": [],  # 穿搭、全身
        "deepfashion2": []  # 单品
    }
    
    for doc, img_id in zip(results["documents"][0], results["ids"][0]):
        for item in all_items:
            if item["image_id"] == img_id:
                if item["source"] == "fashionpedia":
                    categorized_results["fashionpedia"].append(item)
                else:
                    categorized_results["deepfashion2"].append(item)
                break
    
    # 生成建议
    context = "\n".join([f"[{item['source']}] {item['description']}" for item in all_items[:5]])
    
    prompt = f"""
User needs: {user_profile['special_needs']}
Occasion: {user_profile['occasion']}
Style: {user_profile['style']}

Provide outfit recommendation using both Fashionpedia (outfits) and DeepFashion2 (items).
"""
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a fashion stylist."},
                 {"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    return {
        "success": True,
        "advice": response.choices[0].message.content,
        "analysis": analysis,
        "categorized_results": categorized_results,
        "user_profile": user_profile
    }

@app.route('/')
def index():
    return render_template('index_combined.html')

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
        result = rag_query(user_profile)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    print("🚀 Fashion RAG Server (Multi-DB) started at http://localhost:5006")
    app.run(debug=True, port=5006, use_reloader=False)

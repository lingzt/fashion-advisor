"""
Fashion RAG Server - Dual Database (Fashionpedia + DeepFashion2)
"""
import json
import chromadb
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, template_folder='templates', static_folder='.')
CORS(app)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

def load_fashionpedia():
    """加载 Fashionpedia 数据"""
    path = "instances_attributes_val2020.json"
    if not os.path.exists(path):
        return []
    
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
    """加载 DeepFashion2 数据（仅图片，无详细标注）"""
    img_dir = "deepfashion2_test/image"
    if not os.path.exists(img_dir):
        return []
    
    # 从文件名生成简单描述
    category_map = {
        "short_sleeved_shirt": "Short-sleeved shirt",
        "long_sleeved_shirt": "Long-sleeved shirt",
        "short_sleeved_outwear": "Short-sleeved jacket",
        "long_sleeved_outwear": "Long-sleeved coat",
        "vest": "Vest",
        "sling": "Sling top",
        "shorts": "Shorts",
        "trousers": "Trousers",
        "skirt": "Skirt",
        "short_sleeved_dress": "Short-sleeved dress",
        "long_sleeved_dress": "Long-sleeved dress",
        "vest_dress": "Vest dress",
        "sling_dress": "Sling dress"
    }
    
    items = []
    # 取前1000张图片
    for i, filename in enumerate(sorted(os.listdir(img_dir))[:1000]):
        if filename.endswith('.jpg'):
            # 从文件名推断（实际应该从标注获取，这里简化处理）
            items.append({
                "description": f"[DeepFashion2] Fashion item - street style photography",
                "image_id": f"df2_{i}",
                "file_path": f"deepfashion2_test/image/{filename}",
                "category": "fashion",
                "source": "deepfashion2"
            })
    
    return items

def build_index(items, name):
    """构建向量索引"""
    from chromadb.utils import embedding_functions
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
    )
    client = chromadb.Client()
    collection = client.create_collection(name=name, embedding_function=openai_ef)
    
    documents = [item["description"] for item in items]
    ids = [item["image_id"] for item in items]
    collection.add(documents=documents, ids=ids)
    
    return collection, items

@app.route('/')
def index():
    return render_template('index_dual.html')

@app.route('/test/<path:filename>')
def serve_image(filename):
    return send_from_directory('.', f'test/{filename}')

@app.route('/deepfashion2/<path:filename>')
def serve_df2_image(filename):
    return send_from_directory('deepfashion2_test', f'image/{filename}')

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
        query = f"{user_profile['style']} {user_profile['occasion']} {user_profile['special_needs']}"
        
        # 加载 Fashionpedia
        fp_items = load_fashionpedia()
        fp_collection, fp_all = build_index(fp_items, "fashionpedia_rag")
        
        # 加载 DeepFashion2
        df2_items = load_deepfashion2()
        
        # 检索
        results = fp_collection.query(query_texts=[query], n_results=5)
        
        categorized_results = {
            "fashionpedia": [],
            "deepfashion2": []
        }
        
        for doc, img_id in zip(results["documents"][0], results["ids"][0]):
            for item in fp_all:
                if item["image_id"] == img_id:
                    categorized_results["fashionpedia"].append(item)
                    break
        
        # 从 DeepFashion2 取样
        categorized_results["deepfashion2"] = df2_items[:5]
        
        # 生成建议
        context = "\n".join([f"[{item['source']}] {item['description']}" for item in categorized_results["fashionpedia"][:3]])
        
        prompt = f"""
User: {user_profile['identity']}, {user_profile['age']}
Occasion: {user_profile['occasion']} ({user_profile['special_needs']})
Style: {user_profile['style']}

Provide outfit recommendation:
{context}

Reply in English.
"""
        
        import openai
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a fashion stylist."},
                    {"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return jsonify({
            "success": True,
            "result": {
                "advice": response.choices[0].message.content,
                "categorized_results": categorized_results,
                "user_profile": user_profile
            }
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    print("🚀 Fashion RAG Server (Dual DB) started at http://localhost:5006")
    app.run(debug=True, port=5006, use_reloader=False)

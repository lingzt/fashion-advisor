"""
Fashionpedia RAG Demo with GPT-4
"""
import json
import openai
import chromadb
import os

# 配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
openai.api_key = OPENAI_API_KEY

# 1. 加载 Fashionpedia 标注
print("📦 加载标注文件...")
with open("instances_attributes_val2020.json", "r") as f:
    data = json.load(f)

categories = {cat["id"]: cat["name"] for cat in data["categories"]}
attributes = {attr["id"]: attr["name"] for attr in data["attributes"]}

# 2. 创建图片描述
print("🖼️ 生成图片描述...")
image_descriptions = []
image_ids = []

for img in data["images"][:100]:  # 先用100张测试
    annotations = [ann for ann in data["annotations"] if ann["image_id"] == img["id"]]
    if annotations:
        desc_parts = []
        for ann in annotations[:3]:
            cat = categories.get(ann["category_id"], "unknown")
            attrs = [attributes.get(aid, "") for aid in ann.get("attribute_ids", [])]
            desc_parts.append(f"{cat}: {', '.join(attrs)}")
        description = f"图片包含: {'; '.join(desc_parts)}"
        image_descriptions.append(description)
        image_ids.append(img["id"])

print(f"✅ 生成了 {len(image_descriptions)} 个描述")

# 3. 初始化 ChromaDB + OpenAI Embeddings
print("🔧 初始化向量数据库...")
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

client = chromadb.Client()
collection = client.create_collection(
    name="fashionpedia_rag",
    embedding_function=openai_ef
)

# 4. 存入向量数据库
print("💾 存入向量数据库...")
collection.add(
    documents=image_descriptions,
    ids=[str(img_id) for img_id in image_ids]
)
print("✅ 完成!")

# 5. RAG 查询
def rag_query(question: str, n_results: int = 3):
    print(f"\n🔍 问题: {question}")
    
    results = collection.query(query_texts=[question], n_results=n_results)
    retrieved = results["documents"][0]
    
    print(f"📋 检索到 {len(retrieved)} 个相关图片:")
    for i, doc in enumerate(retrieved):
        print(f"   {i+1}. {doc}")
    
    context = "\n".join([f"[图片{i+1}] {doc}" for i, doc in enumerate(retrieved)])
    prompt = f"""
基于以下时尚图片信息回答问题：

{context}

用户问题: {question}
请基于以上图片信息给出建议。
"""
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个时尚顾问，基于检索到的图片信息回答问题。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

# 测试
if __name__ == "__main__":
    while True:
        question = input("\n❓ 请输入问题 (输入 q 退出): ")
        if question.lower() == "q":
            break
        answer = rag_query(question)
        print(f"\n💬 回答: {answer}")

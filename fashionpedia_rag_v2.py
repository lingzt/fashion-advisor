"""
Fashionpedia RAG Demo with GPT-4 + 图片展示
"""
import json
import openai
import chromadb
import os

# 配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
openai.api_key = OPENAI_API_KEY

# 图片目录
IMAGE_DIR = "test"  # Fashionpedia 图片在 test 目录

# 1. 加载 Fashionpedia 标注
print("📦 加载标注文件...")
with open("instances_attributes_val2020.json", "r") as f:
    data = json.load(f)

categories = {cat["id"]: cat["name"] for cat in data["categories"]}
attributes = {attr["id"]: attr["name"] for attr in data["attributes"]}
image_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

# 2. 创建图片描述
print("🖼️ 生成图片描述...")
image_data = []  # 存储 (描述, 图片ID, 文件路径)

for img in data["images"][:500]:  # 用500张测试
    annotations = [ann for ann in data["annotations"] if ann["image_id"] == img["id"]]
    if annotations:
        desc_parts = []
        for ann in annotations[:3]:
            cat = categories.get(ann["category_id"], "unknown")
            attrs = [attributes.get(aid, "") for aid in ann.get("attribute_ids", [])]
            desc_parts.append(f"{cat}: {', '.join(attrs)}")
        description = f"图片包含: {'; '.join(desc_parts)}"
        file_name = img["file_name"]
        image_data.append({
            "description": description,
            "image_id": str(img["id"]),
            "file_path": f"{IMAGE_DIR}/{file_name}"
        })

print(f"✅ 生成了 {len(image_data)} 个描述")

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
documents = [item["description"] for item in image_data]
ids = [item["image_id"] for item in image_data]
collection.add(
    documents=documents,
    ids=ids
)
print("✅ 完成!")

# 5. RAG 查询 + 图片展示
def rag_query_with_images(question: str, n_results: int = 3):
    print(f"\n🔍 问题: {question}")
    print("=" * 60)
    
    # 检索
    results = collection.query(query_texts=[question], n_results=n_results)
    retrieved_ids = results["ids"][0]
    retrieved_docs = results["documents"][0]
    
    print(f"📋 检索到 {len(retrieved_ids)} 个相关图片:\n")
    
    # 查找对应的图片信息
    matched_images = []
    for i, (doc, img_id) in enumerate(zip(retrieved_docs, retrieved_ids)):
        # 找到对应的图片信息
        img_info = next((item for item in image_data if item["image_id"] == img_id), None)
        if img_info:
            matched_images.append(img_info)
            print(f"   【图片 {i+1}】ID: {img_id}")
            print(f"   📝 描述: {doc}")
            print(f"   🖼️  文件: {img_info['file_path']}")
            print()
    
    # 构建 Prompt
    context = "\n".join([f"[图片{i+1}] {doc}" for i, doc in enumerate(retrieved_docs)])
    prompt = f"""
基于以下时尚图片信息回答用户问题：

{context}

用户问题: {question}
请基于以上图片信息给出实用的时尚建议。
"""
    
    # GPT-4 生成
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个专业时尚顾问，基于图片信息给出实用建议。用中文回答。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    answer = response.choices[0].message.content
    print("=" * 60)
    print(f"💬 时尚建议:\n{answer}")
    print("=" * 60)
    
    # 返回图片路径供展示
    return [img["file_path"] for img in matched_images]

# 测试
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 命令行参数模式
        question = " ".join(sys.argv[1:])
        image_paths = rag_query_with_images(question)
        print(f"\n🖼️ 相关图片路径:")
        for path in image_paths:
            print(f"   {path}")
    else:
        # 交互模式
        while True:
            try:
                question = input("\n❓ 请输入问题 (输入 q 退出): ")
                if question.lower() == "q":
                    break
                image_paths = rag_query_with_images(question)
                print(f"\n🖼️ 相关图片路径:")
                for path in image_paths:
                    print(f"   {path}")
            except EOFError:
                break

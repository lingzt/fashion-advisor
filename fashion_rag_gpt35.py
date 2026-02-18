"""
Fashionpedia RAG - GPT-3.5 版本（便宜 + 够用）
"""
import json
import openai
import chromadb
import os

# 配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai.api_key = OPENAI_API_KEY
IMAGE_DIR = "test"

# 用户画像
USER_PROFILE = {
    "年龄": "60岁",
    "身份": "家庭主妇",
    "家庭情况": "有4个小孩",
    "场合": "参加婚礼",
    "主题": "狂野西部 (Wild West)",
    "风格偏好": "优雅得体",
    "特殊需求": "需要方便照顾小孩，舒适优先",
    "初始需求": "参加西部主题婚礼"
}


def rag_query_with_profile(profile, n_results=5):
    """基于用户画像构建查询"""
    
    print("\n" + "="*60)
    print("👗 时尚穿搭顾问 (GPT-3.5-Turbo)")
    print("="*60)
    
    print("\n📋 用户画像:")
    for k, v in profile.items():
        print(f"   • {k}: {v}")
    
    # 加载数据
    print("\n🔄 加载 Fashionpedia 数据...")
    with open("instances_attributes_val2020.json", "r") as f:
        data = json.load(f)
    
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    attributes = {attr["id"]: attr["name"] for attr in data["attributes"]}
    image_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    
    # 生成图片描述
    print("🖼️ 处理图片标注...")
    image_data = []
    for img in data["images"][:500]:
        annotations = [ann for ann in data["annotations"] if ann["image_id"] == img["id"]]
        if annotations:
            desc_parts = []
            for ann in annotations[:3]:
                cat = categories.get(ann["category_id"], "unknown")
                attrs = [attributes.get(aid, "") for aid in ann.get("attribute_ids", [])]
                desc_parts.append(f"{cat}: {', '.join(attrs)}")
            description = f"图片包含: {'; '.join(desc_parts)}"
            image_data.append({
                "description": description,
                "image_id": str(img["id"]),
                "file_path": f"{IMAGE_DIR}/{image_id_to_file.get(img['id'], '')}"
            })
    
    print(f"✅ 生成 {len(image_data)} 条图片描述")
    
    # 构建查询
    query = f"用户: {profile['身份']}, {profile['年龄']} - 场合: {profile['场合']}, 主题: {profile['主题']} - 风格: {profile['风格偏好']} - 需求: {profile['特殊需求']}"
    
    print(f"\n🔍 查询: {query[:80]}...")
    
    # 初始化向量数据库
    print("\n💾 初始化向量数据库...")
    from chromadb.utils import embedding_functions
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
    )
    client = chromadb.Client()
    collection = client.create_collection(name="fashion_rag_gpt35", embedding_function=openai_ef)
    collection.add(
        documents=[item["description"] for item in image_data],
        ids=[item["image_id"] for item in image_data]
    )
    
    # 检索
    print("🔎 检索相关图片...")
    results = collection.query(query_texts=[query], n_results=n_results)
    retrieved_docs = results["documents"][0]
    retrieved_ids = results["ids"][0]
    
    print(f"✅ 检索到 {len(retrieved_docs)} 张图片")
    
    # 显示检索结果
    print("\n📷 检索结果:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"   {i}. {doc[:70]}...")
    
    # GPT-3.5 生成建议（便宜 90%！）
    context = "\n".join([f"[图片{i+1}] {doc}" for i, doc in enumerate(retrieved_docs)])
    prompt = f"""
你是专业时尚顾问。基于以下信息给出穿搭建议：

=== 用户画像 ===
- 年龄: {profile['年龄']}
- 身份: {profile['身份']}（有{profile['家庭情况']}）
- 场合: {profile['场合']}
- 主题: {profile['主题']}
- 风格: {profile['风格偏好']}
- 需求: {profile['特殊需求']}

=== 参考图片 ===
{context}

请给出具体穿搭建议：上装、下装、鞋子、配饰、颜色。
用中文，简洁实用。
"""
    
    print("\n🤖 GPT-3.5 生成穿搭建议...\n")
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # 便宜 90%！
        messages=[
            {"role": "system", "content": "你是专业时尚顾问，简洁实用地回答。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    answer = response.choices[0].message.content
    
    print("="*60)
    print("💬 穿搭建议 (GPT-3.5):")
    print("="*60)
    print(answer)
    print("="*60)
    
    # 返回图片路径
    matched_images = []
    for img_id in retrieved_ids:
        for item in image_data:
            if item["image_id"] == img_id:
                matched_images.append(item["file_path"])
                break
    
    return answer, matched_images


def main():
    answer, images = rag_query_with_profile(USER_PROFILE)
    
    print("\n🖼️ 推荐图片:")
    for i, path in enumerate(images, 1):
        print(f"   {i}. {path}")
    
    print("\n" + "="*60)
    print("💡 成本对比：GPT-3.5 比 GPT-4 便宜 ~90%")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

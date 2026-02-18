"""
Fashionpedia RAG + 对话式信息提取系统
"""
import json
import openai
import chromadb
import os

# 配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
openai.api_key = OPENAI_API_KEY
IMAGE_DIR = "test"

# 信息收集模板
INFO_TEMPLATE = {
    "年龄": None,
    "职业/身份": None,
    "身材特点": None,
    "风格偏好": None,  # 简约/时尚/保守/个性
    "预算": None,  # 低/中/高
    "场合类型": None,  # 婚礼/聚会/工作/日常
    "主题要求": None,  # 如: 西部/复古/正式
    "季节": None,  # 春/夏/秋/冬
    "特殊需求": None,  # 如: 舒适/显瘦/方便活动
}

# 追问问题映射
FOLLOW_UP_QUESTIONS = {
    "年龄": "请问您的年龄是多少呢？",
    "职业/身份": "您的职业或日常身份是什么？（如：家庭主妇/职场人士/学生）",
    "身材特点": "您对自己身材有什么特别关注的地方吗？（如：高/矮/胖/瘦）",
    "风格偏好": "您平时喜欢什么风格的穿着？（简约/时尚/保守/个性/其他）",
    "预算": "您的穿搭预算大概是多少？（低/中/高）",
    "场合类型": "这次要参加什么类型的活动呢？（婚礼/聚会/工作/日常/其他）",
    "主题要求": "活动有什么主题要求吗？（如：正式/休闲/特定主题）",
    "季节": "活动在什么季节举办？（春夏秋冬）",
    "特殊需求": "还有其他特别要求吗？（如：需要照顾小孩要方便活动/过敏/舒适度优先等）",
}

# 对话策略
EXTRACTION_PROMPT = """你是一个专业时尚顾问，正在帮助用户制定穿搭方案。

当前已收集的信息：
{collected_info}

请根据以下策略回应：

1. 如果所有关键信息已收集完整 → 回复 "READY"（并附带完整的用户画像摘要）
2. 如果缺少关键信息 → 礼貌追问一个最相关的问题
3. 如果用户回答了问题 → 确认理解并继续追问下一个缺失信息

关键信息优先级：
1. 年龄
2. 场合类型
3. 风格偏好
4. 主题要求（如有）
5. 身材特点
6. 特殊需求

请直接回复，不要有多余的客套话。
"""


def check_info_complete(info):
    """检查关键信息是否收集完整"""
    required = ["年龄", "场合类型", "风格偏好", "主题要求"]
    missing = [k for k in required if not info.get(k)]
    return len(missing) == 0, missing


def extract_info_step_by_step():
    """引导式信息提取"""
    info = INFO_TEMPLATE.copy()
    
    print("\n" + "="*60)
    print("👗 时尚穿搭顾问")
    print("="*60)
    print("我来帮您找到最合适的穿搭方案！")
    print("请回答几个问题，让我更了解您的需求。\n")
    
    # 初始简短问候
    print("💬 请简单描述您要参加的场合和基本需求：")
    initial_input = input("> ").strip()
    
    # 快速解析初始输入
    info["初始需求"] = initial_input
    
    # 依次询问缺失的关键信息
    for key in ["年龄", "场合类型", "风格偏好", "主题要求", "身材特点", "特殊需求"]:
        if info.get(key) is None:
            print(f"\n{FOLLOW_UP_QUESTIONS[key]}")
            answer = input("> ").strip()
            if answer:
                info[key] = answer
    
    # 检查是否完整
    complete, missing = check_info_complete(info)
    
    # 使用 GPT-4 确认和补充
    collected_str = "\n".join([f"- {k}: {v}" for k, v in info.items() if v])
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个专业时尚顾问，简洁高效地收集用户信息。"},
            {"role": "user", "content": EXTRACTION_PROMPT.format(collected_info=collected_str)}
        ],
        temperature=0.3
    )
    
    return info, response.choices[0].message.content


# RAG 查询函数（复用之前的逻辑）
def rag_query(question: str, n_results: int = 3):
    """RAG 查询"""
    # 加载数据
    with open("instances_attributes_val2020.json", "r") as f:
        data = json.load(f)
    
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    attributes = {attr["id"]: attr["name"] for attr in data["attributes"]}
    image_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    
    # 生成图片描述
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
    
    # 初始化向量数据库
    from chromadb.utils import embedding_functions
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
    )
    client = chromadb.Client()
    collection = client.create_collection(name="fashion_rag", embedding_function=openai_ef)
    collection.add(
        documents=[item["description"] for item in image_data],
        ids=[item["image_id"] for item in image_data]
    )
    
    # 检索
    results = collection.query(query_texts=[question], n_results=n_results)
    retrieved_docs = results["documents"][0]
    retrieved_ids = results["ids"][0]
    
    # GPT-4 生成建议
    context = "\n".join([f"[图片{i+1}] {doc}" for i, doc in enumerate(retrieved_docs)])
    prompt = f"""
基于以下时尚图片信息，为用户制定穿搭方案：

用户需求：{question}

参考图片信息：
{context}

请给出实用的穿搭建议，包括具体单品推荐和搭配要点。
"""
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是专业时尚顾问，给出具体、实用的穿搭建议。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    # 返回结果和图片路径
    matched_images = []
    for img_id in retrieved_ids:
        for item in image_data:
            if item["image_id"] == img_id:
                matched_images.append(item["file_path"])
                break
    
    return response.choices[0].message.content, matched_images


def generate_user_profile(info):
    """生成用户画像摘要"""
    return f"""
=== 用户画像 ===
年龄: {info.get('年龄', '未提供')}
身份: {info.get('职业/身份', '未提供')}
风格偏好: {info.get('风格偏好', '未提供')}
场合: {info.get('场合类型', '未提供')}
主题: {info.get('主题要求', '未提供')}
身材: {info.get('身材特点', '未提供')}
特殊需求: {info.get('特殊需求', '未提供')}
初始需求: {info.get('初始需求', '未提供')}
"""


def main():
    """主流程"""
    # Step 1: 收集信息
    info, gpt_response = extract_info_step_by_step()
    
    print("\n" + "="*60)
    
    # 检查是否准备就绪
    if "READY" in gpt_response:
        print("✅ 信息收集完成！")
        print(generate_user_profile(info))
        
        # Step 2: 构建 RAG 查询
        print("\n🔍 正在为您搜索合适的穿搭方案...\n")
        
        # 构建详细查询
        query_parts = []
        if info.get("年龄"):
            query_parts.append(f"{info['年龄']}岁")
        if info.get("场合类型"):
            query_parts.append(info["场合类型"])
        if info.get("主题要求"):
            query_parts.append(info["theme"])
        if info.get("风格偏好"):
            query_parts.append(info["风格偏好"])
        if info.get("特殊需求"):
            query_parts.append(info["特殊需求"])
        
        query = " ".join(query_parts)
        
        # RAG 查询
        answer, image_paths = rag_query(query)
        
        print("="*60)
        print("💬 穿搭建议:\n")
        print(answer)
        print("\n" + "="*60)
        
        print("\n🖼️ 推荐图片:")
        for i, path in enumerate(image_paths[:3], 1):
            print(f"   {i}. {path}")
        
        print("\n" + "="*60)
        print("💡 如需调整建议或有其他问题，请继续问我！")
        print("="*60 + "\n")
    
    else:
        # 继续追问
        print(f"\n💬 {gpt_response}")
        # 这里可以递归继续对话，为简洁起见，暂时跳过


if __name__ == "__main__":
    main()

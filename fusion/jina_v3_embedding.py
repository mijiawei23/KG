import requests
import json
from time import sleep

# 配置信息
API_URL = ""
HEADERS = {
    "Authorization": "Bearer xiaoyu-embedding",
    "Content-Type": "application/json"
}
MODEL_NAME = "jina-v3"
INPUT_FILE = ""
OUTPUT_FILE = ""


def get_embedding(entity):
    """调用API获取实体向量"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": entity}
        ]
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        response.raise_for_status()

        # 解析响应
        response_data = response.json()
        if response_data.get('data'):
            return response_data['data'][0].get('embedding', [])
        return []

    except requests.exceptions.RequestException as e:
        print(f"请求失败（{entity}）: {str(e)}")
        return None


def process_entities():
    """处理所有实体并保存结果"""
    # 读取JSON文件
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entities_with_types = []

    for item in data:
        # 提取实体和类型
        entity = item.get("entity")
        types = item.get("types", [])

        # 过滤无效的entity
        if not entity or not isinstance(entity, str):
            continue
        entity = entity.strip()
        if not entity or entity.lower() == "null":
            continue

        # 过滤无效的types
        if not isinstance(types, list):
            continue

        # 处理每个类型
        for entity_type in types:
            # 检查类型是否有效
            if not isinstance(entity_type, str):
                continue
            entity_type = entity_type.strip()
            if not entity_type or entity_type.lower() == "null":
                continue

            entities_with_types.append({
                "entity": entity,
                "type": entity_type
            })

    results = []

    for idx, entity_info in enumerate(entities_with_types, 1):
        entity = entity_info["entity"]
        entity_type = entity_info["type"]
        print(f"正在处理 {idx}/{len(entities_with_types)}: [{entity_type}] {entity}")

        embedding = get_embedding(entity)

        if embedding is not None:
            results.append({
                "entity": entity,
                "type": entity_type,
                "vector": embedding
            })

        # 避免频繁请求，根据需要调整间隔
        sleep(0.1)

    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"处理完成！共成功处理 {len(results)} 个实体，结果已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    process_entities()
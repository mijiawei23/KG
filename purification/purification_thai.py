import json
import os
from openai import OpenAI
from tqdm import tqdm

# 配置参数
CONFIG = {
    "input_file": "D:/学术/datasets/thai/thai_data_purified.json",
    "output_file": "D:/学术/datasets/thai/thai_data_purified.json",
    "allowed_relations": ["cooperation", "lawsuit", "investment", "acquisition", "branch",
        "legal_representative", "executive", "shareholder",
        "registered_address", "branch_address", "work_address",
        "belong", "participate"],
    "model_name": "deepseek-chat",
    "api_key": "",
    "base_url": "",
    "save_interval": 100  # 新增保存间隔
}

SYSTEM_PROMPT = f"""
你是一名多语言产业数据实体关系提纯专家，现在要对三元组进行提纯，给定三元组和实体以及文本，把三元组和实体进行提纯，输出提纯后的三元组和实体,将错误的三元组去除，保留正确的三元组。
提纯规则：
1.实体类型要求：
enterprise（企业）：公司全称，如【ธนาคาร ซีไอเอ็ม ไทย】
person（人物）：包含完整姓名，如【อมรเทพ จาวะลา】
location（地点）：地理名称，如【กรุงเทพฯ】
project（项目）：投资/合作项目名称
去除实体中的指代问题，要求实体是真实的实体。消除别名、缩写和全称的差异。
2. 严格关系类型限制：{CONFIG["allowed_relations"]}
    关系必须满足：
   - 主语和宾语实体类型匹配（如executive关系需连接person→enterprise）
   - 具有现实商业合理性
   - 在文本中有明确依据
3.输出结果必须严格按照以下JSON格式输出：{{
    "purified_entities": {{
        "enterprise": [],
        "person": [],
        "location": [],
        "project": []
    }},
    "purified_triples": [
        "(enterprise, relationship, enterprise)"
    ]
}}
4.实体和三元组要求是符合真实世界事实的，
5. 禁止包含：缩写/别称
   - 不确定的关系
   - 文本中未明确提及的信息
"""

client = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])


def purify_entities(article):
    """实体关系提纯核心函数"""
    # 处理数据类型异常
    entity_relationship = article.get('entity_relationship', {})

    # 处理字符串类型的entity_relationship
    if isinstance(entity_relationship, str):
        try:
            entity_relationship = json.loads(entity_relationship)
        except json.JSONDecodeError:
            entity_relationship = {"entities": [], "triplet": []}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""
        文本内容：{article['content']}
        原始实体：{json.dumps(entity_relationship.get('entities', []), ensure_ascii=False)}
        原始三元组：{entity_relationship.get('triplet', [])}
        请按照要求输出提纯后的实体和三元组：
        """}
    ]

    for _ in range(3):  # 重试机制
        try:
            response = client.chat.completions.create(
                model=CONFIG["model_name"],
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(f"API Error: {str(e)}")
            break
    return None


def save_data(data, file_path):
    """保存数据到指定文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_articles():
    """主处理流程"""
    # 读取输入文件
    with open(CONFIG["input_file"], 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 数据预处理：确保entity_relationship为字典类型
    for article in data:
        er = article.get('entity_relationship')
        if er and isinstance(er, str):
            try:
                article['entity_relationship'] = json.loads(er)
            except json.JSONDecodeError:
                article['entity_relationship'] = {"entities": [], "triplet": []}

    processed_count = 0
    temp_data = []
    pbar = tqdm(data, desc="Processing Articles", unit="article")

    for idx, article in enumerate(pbar):
        # 跳过已处理文章
        if 'purified_entities' in article and 'purified_triples' in article:
            tqdm.write(f" 跳过已处理文章 {article['article_id']}")
            continue

        pbar.set_postfix({"article_id": article["article_id"]})

        # 类型安全检查
        if not isinstance(article.get('entity_relationship'), dict):
            article['entity_relationship'] = {"entities": [], "triplet": []}

        # 执行提纯
        purified = purify_entities(article)

        if purified:
            # 合并结果
            article.update({
                "purified_entities": purified["purified_entities"],
                "purified_triples": purified["purified_triples"]
            })

            # 输出对比信息
            tqdm.write(f"\n Article {article['article_id']}")
            tqdm.write(" 实体对比:")
            tqdm.write(
                f"原始实体: {json.dumps(article['entity_relationship']['entities'], ensure_ascii=False, indent=2)}")
            tqdm.write(f"提纯实体: {json.dumps(purified['purified_entities'], ensure_ascii=False, indent=2)}")

            tqdm.write(" 三元组对比:")
            original_triples = '\n'.join([f" - {t}" for t in article['entity_relationship']['triplet']]) or "无"
            purified_triples = '\n'.join([f" + {t}" for t in purified['purified_triples']]) or "无"
            tqdm.write(f"原始三元组:\n{original_triples}")
            tqdm.write(f"提纯三元组:\n{purified_triples}")
            tqdm.write("─" * 50)
        else:
            article.update({
                "purified_entities": {"enterprise": [], "person": [], "location": [], "project": []},
                "purified_triples": []
            })

        temp_data.append(article)
        processed_count += 1

        # 定期保存
        if processed_count % CONFIG["save_interval"] == 0:
            data[idx - len(temp_data) + 1: idx + 1] = temp_data
            save_data(data, CONFIG["output_file"])
            temp_data = []

    # 保存剩余数据
    if temp_data:
        data[len(data) - len(temp_data):] = temp_data
        save_data(data, CONFIG["output_file"])

    print(f"\n🎉 处理完成！最终结果已保存至 {CONFIG['output_file']}")


if __name__ == "__main__":
    process_articles()
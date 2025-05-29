#coding:utf-8
import json
import os
import re
from collections import defaultdict
from openai import OpenAI

# 环境配置
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""


# 初始化客户端
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)


# 配置参数
CONFIG = {
    "input_file": "",
    "output_file": "",
    "sampling_times": 5,
    "consistency_threshold": 3,
    "allowed_relations": {
        'cooperation', 'lawsuit', 'investment', 'acquisition', 'branch',
        'legal_representative', 'executive', 'shareholder',
        'registered_address', 'branch_address', 'work_address',
        'belong', 'participate'
    }
}

# 示例
example = f"""
    "(SJC,branch,null)":The triplet contains null elements, and the triplet identification error o ccurs
    Correct answer: [triplet:null]
     "(SJC,work_address,Hà Nội)",The triple contains the ticker symbol, which is not an entity
    Correct answer: [triplet:null]
    """
# 输出格式

Output_format="""
    "original_entities": {
        "enterprise": ["浩博医药", "汉康资本", "鼎晖VGC"],
        "person": [],
        "location": [],
        "project": ["B轮融资"]
    },
    "purified_entities": {
        "enterprise": ["浩博医药", "汉康资本"],
        "person": [],
        "location": [],
        "project": ["B轮融资"]
    },
    "original_triples": [
        "(汉康资本, investment, 鼎晖VGC)",
        "(浩博医药, belong, B轮融资)"
    ],
    "purified_triples": [
        "(浩博医药, belong, B轮融资)"
    ]
"""


# 系统提示模板
SYSTEM_PROMPT = f"""
    Given text and triples, extract the correct triples.
    **Examples :**
    {example}
    Step 1: Determine whether the extracted entity matches the facts, such as
    Step 2: Determine whether the relationship is correct,
    Step 3: Extract the correct entities and triples
    Step 4: Strictly follow the given format output, only need to output the answer.
    There can be no reference problem in the entity, such as "company", "he", "张某某",and so on, need to be specific to the entity name，and  triple cannot contain an empty element.
"""

def validate_triple(triple):
    """验证三元组格式有效性"""
    pattern = r"\(([^,]+?),\s*([^,]+?),\s*([^)]+?)\)"
    if not re.fullmatch(pattern, triple):
        return False
    _, relation, _ = re.findall(pattern, triple)[0]
    return relation in CONFIG["allowed_relations"]

def parse_entities_from_response(text):
    """从API响应文本中提取实体"""
    entities = {
        "enterprise": [],
        "person": [],
        "location": [],
        "project": []
    }
    # 假设API返回的实体是类似以下格式：
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("enterprise:"):
            entities["enterprise"] = line.split(":")[1].split(",")
        elif line.startswith("person:"):
            entities["person"] = line.split(":")[1].split(",")
        elif line.startswith("location:"):
            entities["location"] = line.split(":")[1].split(",")
        elif line.startswith("project:"):
            entities["project"] = line.split(":")[1].split(",")
    return entities

def process_article(article):
    """处理单篇文章"""
    result = {
        "purified_entities": {},
        "purified_triples": []
    }

    try:
        content = article.get("content", "")
        original_entities = article.get("entities", {})
        original_triplets = article.get("triplet", [])

        print("原始实体:")
        print(json.dumps(original_entities, ensure_ascii=False, indent=2))
        print("原始三元组:")
        print(json.dumps(original_triplets, ensure_ascii=False, indent=2))

        triples_counter = defaultdict(int)
        entities_counter = defaultdict(lambda: defaultdict(int))

        for _ in range(CONFIG["sampling_times"]):
            try:
                response = client.chat.completions.create(
                    model="ep-20240926204940-gh2p7",
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ]
                )

                new_triples = []
                for line in response.choices[0].message.content.split("\n"):
                    line = line.strip()
                    if validate_triple(line):
                        new_triples.append(line)

                new_entities = parse_entities_from_response(response.choices[0].message.content)

                for triple in set(original_triplets + new_triples):
                    if validate_triple(triple):
                        triples_counter[triple] += 1

                for entity_type in original_entities.keys() | new_entities.keys():
                    original_ents = original_entities.get(entity_type, [])
                    new_ents = new_entities.get(entity_type, [])
                    for entity in original_ents + new_ents:
                        if entity.strip():
                            entities_counter[entity_type][entity.strip()] += 1

            except Exception as e:
                print(f"API调用失败: {str(e)}")

        result["purified_triples"] = [
            triple for triple, count in triples_counter.items()
            if count >= CONFIG["consistency_threshold"]
        ]

        purified_entities = {}
        for entity_type, counter in entities_counter.items():
            purified_entities[entity_type] = [
                entity for entity, count in counter.items()
                if count >= CONFIG["consistency_threshold"]
            ]
        result["purified_entities"] = purified_entities

        print("提纯后的实体:")
        print(json.dumps(result["purified_entities"], ensure_ascii=False, indent=2))
        print("提纯后的三元组:")
        print(json.dumps(result["purified_triples"], ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"处理文章 {article.get('aid', '未知')} 时发生错误: {str(e)}")

    return result

def main():
    # 优先从输出文件加载数据
    if os.path.exists(CONFIG["output_file"]):
        print(f"从输出文件加载数据: {CONFIG['output_file']}")
        with open(CONFIG["output_file"], "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        print(f"从输入文件加载数据: {CONFIG['input_file']}")
        with open(CONFIG["input_file"], "r", encoding="utf-8") as f:
            dataset = json.load(f)

    # 处理数据
    for idx, article in enumerate(dataset):
        try:
            print(f"正在处理文章 {idx + 1}/{len(dataset)}，aid: {article['aid']}")

            if "purified_triples" in article:
                print(f"跳过已处理文章: {article['aid']}")
                continue

            result = process_article(article)
            article.update(result)

            # 每处理50条保存一次
            if (idx + 1) % 50 == 0:
                print(f"处理进度: {idx + 1}/{len(dataset)}")
                with open(CONFIG["output_file"], "w", encoding="utf-8") as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"处理文章 {article['aid']} 时发生错误: {str(e)}")
            continue

    # 最终保存
    with open(CONFIG["output_file"], "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print("处理完成，结果已保存")

if __name__ == "__main__":
    main()
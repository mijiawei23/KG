import json
from openai import AsyncOpenAI
import asyncio
from typing import Dict, List, Tuple
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm.asyncio import tqdm_asyncio

# 配置客户端
client = AsyncOpenAI(
    base_url="",
    api_key=""
)

# 全局配置
MODEL_NAME = "deepseek-chat"
MAX_CONCURRENT_REQUESTS = 30
BATCH_SIZE = 5
RETRY_TIMES = 2
TYPE_MATCH_THRESHOLD = 0.8

def build_alignment_prompt(source_entity: str, entity_type: str, candidates: List[Dict], lang_pair: str) -> str:

    lang_config = {
        "zh->vi": {"source_lang": "中文", "target_lang": "越南语"}
    }
    config = lang_config[lang_pair]
    source_lang = config["source_lang"]
    target_lang = config["target_lang"]

    candidate_list = "\n".join([
        f"{idx + 1}. {item['entity']} (类型:{item['type']} 相似度:{item['similarity']:.4f})"
        for idx, item in enumerate(candidates)
    ])

    return f"""请严格验证以下企业实体对齐：

源实体（{source_lang}）：【{source_entity}】类型：{entity_type}
目标语言：{target_lang}
候选列表：
{candidate_list}

验证步骤：
1. 翻译验证:
是否是官方翻译/标准音译
排除发音相近的不相关单词
2. 语义验证:
实体类型必须完全相同（企业/人员/地点/项目）
核心属性必须匹配（域、函数等）
3. 输出要求:
只返回确定匹配的三元组（源实体，“equal”，目标实体）
必须满足完美的对应关系
禁止随机匹配

输出要求：
仅返回JSON格式：{{"matches": [["原实体", "equal", "目标实体"]]}}"""

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(RETRY_TIMES))
async def api_request(prompt: str) -> str:
    """适配新版SDK的API请求"""
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API请求失败: {str(e)}")
        raise

async def batch_process(prompts: List[str]) -> List[str]:
    """批量处理多个提示"""
    tasks = [api_request(prompt) for prompt in prompts]
    return await tqdm_asyncio.gather(*tasks, desc="批量请求", leave=False)

def filter_candidates(source_type: str, candidates: List[Dict]) -> List[Dict]:
    """候选实体预处理"""
    return [
        candidate for candidate in candidates
        if candidate["type"] == source_type
           and candidate["similarity"] >= TYPE_MATCH_THRESHOLD
           and not any(c.isdigit() for c in candidate["entity"])
    ]

def parse_response(response: str) -> List[Tuple]:
    """解析响应，添加错误日志"""
    try:
        json_str = response[response.find('{'): response.rfind('}') + 1]
        data = json.loads(json_str)
        return [
            tuple(item) for item in data.get("matches", [])
            if len(item) == 3 and item[1] == "equal"
        ]
    except Exception as e:
        print(f"解析失败: {str(e)}")
        print(f"原始响应内容:\n{response}")
        return []

async def process_lang_pair(source_entity: str, source_type: str, candidates: List[Dict], lang_pair: str) -> List[Tuple]:
    """处理单个语言方向（仅zh->vi）"""
    filtered = filter_candidates(source_type, candidates)
    if not filtered:
        return []

    try:
        prompt = build_alignment_prompt(source_entity, source_type, filtered, lang_pair)
        response = await api_request(prompt)

        print(f"\n=== 模型原始输出 ===")
        print(f"源实体: {source_entity}")
        print(f"语言对: {lang_pair}")
        print(f"响应内容:\n{response}\n")

        return parse_response(response)
    except Exception as e:
        print(f"处理失败: {source_entity} {lang_pair} - {str(e)}")
        return []

async def process_entity_batch(entity_batch: List[Tuple], semaphore: asyncio.Semaphore) -> List[Tuple]:

    async with semaphore:
        all_tasks = []
        for entity_name, entity_data in entity_batch:
            task = process_lang_pair(
                entity_name,
                entity_data["type"],
                entity_data["matches"].get("zh->vi", []),
                "zh->vi"
            )
            all_tasks.append(task)

        responses = await asyncio.gather(*all_tasks)
        return [item for sublist in responses for item in sublist]

async def main(input_file: str, output_file: str):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    items = list(data.items())
    batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]

    results = []
    batch_tasks = [process_entity_batch(batch, semaphore) for batch in batches]

    for future in tqdm_asyncio.as_completed(batch_tasks, total=len(batch_tasks), desc="批量进度"):
        try:
            batch_result = await future
            results.extend(batch_result)
        except Exception as e:
            print(f"批量处理异常: {str(e)[:100]}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main("",
                     ""))
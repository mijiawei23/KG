import json
import os
import time

import requests
from typing import Dict, List, Any

DEEPSEEK_URL = ""
API_KEY = ""
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

LANG_MAP = {
    "vi": "越南语",
    "th": "泰语",
    "zh": "中文"
}

SYSTEM_PROMPT = """您是多语言实体对齐专家，请严格按以下规则验证实体对：
1. 翻译验证：是否为官方翻译/标准音译
2. 语义验证：类型一致、核心属性匹配
3. 只返回完全匹配的实体对

输出格式：{"matches": [["原实体", "equal", "目标实体"]]}"""


def build_prompt(source_entity: str, entity_type: str, candidates: List[Dict], target_lang: str) -> str:
    candidate_desc = "\n".join(
        [f"{i + 1}. {c['entity']} (相似度: {c['similarity']:.4f})"
         for i, c in enumerate(candidates)]
    )
    return f"""源实体：{source_entity}（类型：{entity_type}）
目标语言：{LANG_MAP[target_lang]}
候选列表：
{candidate_desc}

请验证并输出确认为同一实体的匹配对："""


def call_deepseek(prompt: str) -> List[List[str]]:
    try:
        response = requests.post(
            DEEPSEEK_URL,
            headers=HEADERS,
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1
            },
            timeout=30
        )
        if response.status_code == 200:
            result = json.loads(response.json()["choices"][0]["message"]["content"])
            return result.get("matches", [])
    except Exception as e:
        print(f"API调用失败: {str(e)}")
    return []


def process_entities(input_file: str, output_file: str, batch_size: int = 100):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    batch_count = 0
    start_time = time.time()  # 记录开始时间
    total_entities = len(data)
    processed_pairs = 0  # 新增处理计数器

    print(f"🟢 开始处理，共 {total_entities} 个实体需要处理")

    for idx, (source_entity, entity_info) in enumerate(data.items()):
        entity_type = entity_info["type"]

        # 新增语言对统计
        lang_pairs = [lp for lp in ["zh->vi", "zh->th", "vi->th"] if lp in entity_info["matches"]]
        print(f"\n🔵 处理第 {idx + 1}/{total_entities} 个实体: {source_entity}")
        print(f"   📌 需要处理的语言对: {len(lang_pairs)} 个")

        for lang_idx, lang_pair in enumerate(lang_pairs):
            target_lang = lang_pair.split("->")[-1]
            candidates = entity_info["matches"][lang_pair]
            processed_pairs += 1  # 更新计数器

            # 新增候选数量显示
            print(f"   🌐 处理语言对 ({lang_idx + 1}/{len(lang_pairs)}) {lang_pair}")
            print(f"      📋 候选数量: {len(candidates)} | 已用时间: {time.time() - start_time:.1f}s")

            prompt = build_prompt(source_entity, entity_type, candidates, target_lang)
            matches = call_deepseek(prompt)

            # 新增匹配结果统计
            if matches:
                valid_matches = [m for m in matches if m[1] == "equal"]
                print(f"      ✅ 发现有效匹配: {len(valid_matches)} 条")
                results.extend(valid_matches)
            else:
                print("      ⚠️ 未找到有效匹配")

            # 新增进度预测
            avg_time = (time.time() - start_time) / processed_pairs if processed_pairs else 0
            remaining = avg_time * (total_entities * 3 - processed_pairs)
            print(f"      ⏱️ 预计剩余时间: {remaining / 60:.1f} 分钟")

        # 保存批次时增加详细输出
        if len(results) >= batch_size:
            save_batch(output_file, results, batch_count)
            batch_count += 1
            results = []
            print(f"\n🔶 已处理 {processed_pairs} 个语言对")
            print(f"🕒 平均速度: {processed_pairs / (time.time() - start_time):.1f} 对/秒")

    # 最终保存和统计
    if results:
        save_batch(output_file, results, batch_count)

    # 新增最终统计
    total_time = time.time() - start_time
    print("\n🎉 处理完成！最终统计:")
    print(f"  总计处理实体: {total_entities} 个")
    print(f"  处理语言对: {processed_pairs} 对")
    print(f"  发现匹配项: {len(results)} 条")
    print(f"  生成批次文件: {batch_count + 1 if results else batch_count} 个")
    print(f"  总耗时: {total_time / 60:.1f} 分钟")
    print(f"  平均速度: {processed_pairs / total_time:.1f} 对/秒")


def save_batch(output_file: str, results: List, batch_num: int):
    filename = f"{output_file}_batch_{batch_num}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 保存批次 {batch_num} -> {os.path.abspath(filename)}")
    print(f"   本批次包含 {len(results)} 条对齐结果")


if __name__ == "__main__":
    process_entities(
        input_file="",
        output_file="",
        batch_size=100
    )
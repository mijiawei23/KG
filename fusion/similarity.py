import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os


def load_entities(file_path):
    """加载实体数据并按类型分组+归一化处理"""
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_entities = json.load(f)

    # 按类型分组
    type_groups = defaultdict(list)
    for ent in raw_entities:
        type_groups[ent['type']].append(ent)

    # 对每个类型进行归一化处理
    processed = {}
    for ent_type, entities in type_groups.items():
        vectors = [e['vector'] for e in entities]
        matrix = np.array(vectors, dtype=np.float32)

        # L2归一化
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix_normalized = matrix / norms

        processed[ent_type] = {
            'matrix': matrix_normalized,
            'names': [e['entity'] for e in entities]
        }

    return {
        'all_entities': raw_entities,  # 保留原始数据用于遍历
        'by_type': processed  # 归一化后的分类数据
    }


def main():
    # 配置参数
    paths = {
        'zh': '',
        'vi': '',
        'th': ''
    }
    output_dir = ''  # 输出目录
    similarity_threshold = 0.7

    # 需要处理的语言对组合及文件名映射
    ALLOWED_PAIRS = [
        ('zh', 'vi'),
        ('zh', 'th'),
        ('vi', 'th')
    ]
    lang_display_map = {
        'zh': 'zh',
        'vi': 'vi',
        'th': 'thai'
    }

    # 加载所有数据
    print("Loading data...")
    lang_data = {lang: load_entities(path) for lang, path in paths.items()}

    # 处理每个语言对
    for src_lang, tgt_lang in ALLOWED_PAIRS:
        print(f"\nProcessing language pair: {src_lang}->{tgt_lang}")
        results = {}

        # 获取源语言的所有实体
        src_entities = lang_data[src_lang]['all_entities']
        # 获取目标语言的类型数据
        tgt_type_data = lang_data[tgt_lang]['by_type']

        # 遍历源实体
        for src_ent in tqdm(src_entities, desc=f"{src_lang}->{tgt_lang}"):
            src_type = src_ent['type']
            # 跳过目标语言中没有的类型
            if src_type not in tgt_type_data:
                continue

            # 源向量归一化
            src_vector = np.array(src_ent['vector'], dtype=np.float32)
            src_norm = np.linalg.norm(src_vector)
            if src_norm == 0:
                src_vector_normalized = src_vector
            else:
                src_vector_normalized = src_vector / src_norm

            # 获取目标语言同类型数据
            tgt_matrix = tgt_type_data[src_type]['matrix']
            tgt_names = tgt_type_data[src_type]['names']

            # 计算余弦相似度
            cosine_sim = np.dot(tgt_matrix, src_vector_normalized)

            # 取Top-10并过滤
            top_indices = np.argpartition(-cosine_sim, 10)[:10]
            top_sim = cosine_sim[top_indices]

            # 精确排序
            sorted_order = np.argsort(-top_sim)
            top_indices = top_indices[sorted_order]
            top_sim = top_sim[sorted_order]

            # 构建匹配结果
            matches = []
            for idx, sim in zip(top_indices, top_sim):
                if sim < similarity_threshold:
                    break
                matches.append({
                    "entity": tgt_names[idx],
                    "similarity": float(sim),
                    "type": src_type
                })

            # 保存匹配结果
            if matches:
                pair_key = f"{src_lang}->{tgt_lang}"
                results[src_ent['entity']] = {
                    "type": src_type,
                    "matches": {pair_key: matches}
                }

        # 生成输出路径
        src_display = lang_display_map[src_lang]
        tgt_display = lang_display_map[tgt_lang]
        output_path = os.path.join(output_dir, f"{src_display}-{tgt_display}.json")

        # 保存JSON文件
        print(f"Saving results to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
import json
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

EXAMPLE_LIB_PATH = '' 
os.environ["OPENAI_API_KEY"] = ""
# 设置 OPENAI_BASE_URL 环境变量
os.environ["OPENAI_BASE_URL"] = ""

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)


class ExampleSelector:
    def __init__(self, example_path, k=3):
        with open(example_path, 'r', encoding='utf-8') as f:
            self.examples = json.load(f)

        # 准备TF-IDF特征
        self.vectorizer = TfidfVectorizer()
        self.knn = NearestNeighbors(n_neighbors=k, metric='cosine')

        # 提取文本内容并训练模型
        self.example_texts = [ex['content'] for ex in self.examples]
        self._train_model()

    def _train_model(self):
        X = self.vectorizer.fit_transform(self.example_texts)
        self.knn.fit(X)

    def get_similar_examples(self, query_text):
        query_vec = self.vectorizer.transform([query_text])
        distances, indices = self.knn.kneighbors(query_vec)

        selected_examples = []
        for idx in indices[0]:
            example = self.examples[idx]
            selected_examples.append({
                "content": example['content'],
                "answer": json.dumps(example['answer'], ensure_ascii=False)
            })
        return selected_examples


def build_dynamic_prompt(examples):
    example_section = "\n\n## 参考示例："
    for i, ex in enumerate(examples, 1):
        example_section += f"""
示例 {i}:
输入内容：{ex['content']}
输出结果：{ex['answer']}"""
    return example_section


# 初始化示例选择器
example_selector = ExampleSelector(EXAMPLE_LIB_PATH, k=3)

# 主处理逻辑
with open('D:/学术/datasets/thai/thai_data.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# 基础提示模板
base_prompt = """
Task: 你是产业领域的实体关系抽取专家，按照以下步骤来从文本中抽取出实体关系。

Step 1 - 实体抽取
严格按照以下的实体类型从文本中抽取实体：
- 企业 (企业组织，要求是全称)
- 人物 (与企业相关的人物)
- 地点 (与企业相关的地点)
- 项目 (企业所涉及到的项目)

Step 2 - 关系三元组抽取
第一步抽取实体完成之后，根据抽取的实体以及以下的关系类型来从文本中抽取出关系三元组,要求关系是英文的，
[Enterprise]-[Enterprise]:
  • Cooperation (企业之间的合作)
  • Litigation (企业在经营活动中，因合同纠纷、知识产权侵权、劳动争议等法律问题而向法院提起的诉讼。)
  • Investment (向公司或其他企业实体出资，以获取股权或债权等投资回报的关系)
  • Acquired (取得某一企业的部分或全部所有权的收购关系)
  • Branch (公司经营过程中，因为业务需要依法设立的相对独立的分支机构。)

[Person]-[Enterprise]:
  • Legal_representative (依照法律或者法人组织章程规定，代表法人行使职权的公司负责人)
  • Executive (具有企业相应的管理权力和责任的人，如公司的经理、副经理、财务负责人)
  • Litigant (企业在经营活动中，因合同纠纷、知识产权侵权、劳动争议等法律问题而向法院提起的诉讼。，针对企业与个人之间)
  • Shareholder (并凭持有股票享受股息和红利的个人)

[Enterprise]-[Location]:
  • Registered_address (公司营业执照上登记的住址或地址)
  • Branch_address (企业的分公司的地址，通常是企业在不同地区设立的分支机构的地址。)
  • Work_address (企业的主要办公地址，通常是企业的总部或主要经营场所。)

[Enterprise]-[Project]:
  • Belong (属于关系，要求是企业对项目的所有权关系)
  • Investment (企业与项目投资关系，企业对项目的投资)
  • Participation (企业与项目的参与关系，如企业共同参与项目)

Step 3 - 实体关系抽取规则  
1. 实体要求是现实存在的事物，且与实体类型相匹配。关系要求是文中出现与关系类型相关的关键词，或者是可推理出的隐式关系。
2. 禁止输出文中未出现的实体以及关系，若为隐式关系，要为可推理出的。
3. 关系抽取出之后，按照给定的关系类型进行匹配，要为英文的。
4. 

Step 4 - 输出格式校验 
1. 只需要输出答案，不输出其他格式和提示词。
2. 严格按照示例的输出格式输出，实体和关系的输出格式为： entities": {
            "enterprise": [],
            "person": [],
            "location": [],
            "project": []
          },
          "triplet": [
          ]
3. 输出不要包含特殊符号，如（* \/+-`）
以下是类似例子参考：

"""

# 数据处理流程
total_articles = len(json_data)
processed_count = 0

for i, article in enumerate(json_data):
    if "entity_relationship" in article:
        print(f"跳过文章 {article['article_id']}")
        processed_count += 1
        continue

    content = article["content"]
    article_id = article["article_id"]

    # 动态选择示例
    selected_examples = example_selector.get_similar_examples(content)
    dynamic_prompt = base_prompt + build_dynamic_prompt(selected_examples)

    print(f"处理 {i + 1}/{total_articles}: {article_id}")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0,
            messages=[
                {"role": "system", "content": dynamic_prompt},
                {"role": "user", "content": f"请从以下文本中抽取实体关系：\n{content}"}
            ],
            stream=False
        )

        result = response.choices[0].message.content
        article["entity_relationship"] = result

        # 打印处理结果
        print(f"内容: {content}")
        print(f"结果: {result}")
        print("-" * 50)

        processed_count += 1

    except Exception as e:
        print(f"处理失败 {article_id}: {e}")
        article["entity_relationship"] = "ERROR"

    # 定期保存
    if (i + 1) % 2 == 0:
        print(f"保存进度 {i + 1}...")
        try:
            with open('', 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存失败: {e}")

# 最终保存
try:
    with open('', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    print(f"处理完成，保存至 ")
except Exception as e:
    print(f"最终保存失败: {e}")
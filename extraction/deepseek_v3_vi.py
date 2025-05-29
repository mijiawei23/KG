import json
from openai import OpenAI


# 初始化 OpenAI 客户端
client = OpenAI(api_key="", base_url="")

# 读取 JSON 文件
with open('', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# 定义示例
examples = """ 
    example content:阿里巴巴集团宣布，已收购银泰商业集团74%的股份，进一步加强其在零售业的布局。阿里巴巴集团CEO张勇表示，此次收购将有助于集团实现线上线下融合的战略目标。
    银泰商业集团总部位于杭州，旗下拥有多个购物中心和百货公司。
    example answer: [
                    enterprise:阿里巴巴集团, 银泰商业集团,
                    person: 张勇,
                    location:杭州,
                    project:null,
                    triplet:(阿里巴巴集团, acquired, 银泰商业集团),(阿里巴巴集团, executive, 张勇),(银泰商业集团, Registered address, 杭州)
                    ]
    example content:VinGroup đã đầu tư vào dự án 'Smart City' tại thành phố Đà Nẵng, nhằm phát triển các giải pháp công nghệ thông minh cho đô thị. Ông Phạm Nhật Vượng, Chủ tịch của VinGroup, đã công bố kế hoạch này. Dự án 'Smart City' sẽ được quản lý bởi Công ty VinSmart, một đơn vị con của VinGroup. 
    Ngoài ra, VinGroup đã kiện Công ty Công nghệ FutureTech về việc vi phạm hợp đồng cung cấp thiết bị.  
    example answer: [
                    enterprise:VinGroup,Công ty VinSmart,Công ty Công nghệ FutureTech,
                    person: Phạm Nhật Vượng,
                    location:Đà Nẵng,
                    project:Dự án 'Smart City',
                    triplet:(VinGroup investment Dự án 'Smart City'),(VinGroup legal representative Phạm Nhật Vượng),(VinGroup branch Công ty VinSmart),(VinGroup litigation Công ty Công nghệ FutureTech),(Dự án 'Smart City' work address Đà Nẵng)
                    ]                            
"""

# 定义任务提示
taskprompt = f"""
    Task: Extract entities and their relationships from a given text. Extract according to the given entity relationship type and output according to the output format
    Entity type: enterprise, person, location, project.
    Type of relationship: enterprise and enterprise: Partnership, litigation, investment, acquired, branch
    People and enterprise: legal representatives, administrative personnel, litigants, shareholders
    enterprise and location: registered address, branch address, work address.
    enterprise and project: belong, investment, participation.
    Output format: [entity type: entity, triplet:(entity, relationship, entity)]
    There are two examples of the input text and the output format:{examples}
    If there is no entity relationship, output [null].
    """

# 遍历 JSON 数据中的每篇文章
for article in json_data:
    # 如果文章已经包含 'entity_relationship' 字段，跳过处理
    if "entity_relationship" in article:
        print(f"Skipping article {article['aid']}, already processed.")
        continue
    content = article["content"]  # 提取文章内容
    aid = article["aid"]  # 提取文章 ID（可选）

    print(f"Processing article: {aid}")

    # 调用 Deepseek API 处理文章内容
    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0,
        messages=[
            {"role": "system", "content": taskprompt},
            {"role": "user", "content": content},
        ],
        stream=False
    )

    # 提取 API 返回的结果
    result = response.choices[0].message.content

    # 将结果保存到 JSON 数据中（可选）
    article["entity_relationship"] = result

    # 打印结果
    print(f"Article ID: {aid}")
    print(f"Content: {content}")
    print(f"Entity Relationship: {result}")
    print("-" * 50)

# 将更新后的 JSON 数据保存到文件（可选）
with open('', 'w', encoding='utf-8') as file:
    json.dump(json_data, file, ensure_ascii=False, indent=4)

print("Processing complete. Results saved to ''.")
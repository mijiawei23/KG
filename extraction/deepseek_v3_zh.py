import json
import os
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(api_key="", base_url="")

# 读取 JSON 文件
input_file = ''
output_file = ''

try:
    with open(input_file, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
except Exception as e:
    print(f"读取 JSON 文件时出错: {e}")  # 使用 print 输出错误信息
    raise

# 定义示例
examples = """ 
    example content:阿里巴巴集团宣布，已收购银泰商业集团74%的股份，进一步加强其在零售业的布局。阿里巴巴集团CEO张勇表示，此次收购将有助于集团实现线上线下融合的战略目标。
    银泰商业集团总部位于杭州，旗下拥有多个购物中心和百货公司。
    example answer: [enterprise:阿里巴巴集团, 银泰商业集团,
                    person: 张勇,
                    location:杭州,
                    project:null,
                    triplet:(阿里巴巴集团, acquired, 银泰商业集团),(阿里巴巴集团, executive, 张勇),(银泰商业集团, Registered address, 杭州)]
    example content:VinGroup đã đầu tư vào dự án 'Smart City' tại thành phố Đà Nẵng, nhằm phát triển các giải pháp công nghệ thông minh cho đô thị。 Ông Phạm Nhật Vượng, Chủ tịch của VinGroup, đã công bố kế hoạch này. Dự án 'Smart City' sẽ được quản lý bởi Công ty VinSmart, một đơn vị con của VinGroup. 
    Ngoài ra, VinGroup đã kiện Công ty Công nghệ FutureTech về việc vi phạm hợp đồng cung cấp thiết bị.  
    example answer: [enterprise:VinGroup,Công ty VinSmart,Công ty Công nghệ FutureTech,
                    person: Phạm Nhật Vượng,
                    location:Đà Nẵng,
                    project:Dự án 'Smart City',
                    triplet:(VinGroup investment Dự án 'Smart City'),(VinGroup legal representative Phạm Nhật Vượng),(VinGroup branch Công ty VinSmart),(VinGroup litigation Công ty Công nghệ FutureTech),(Dự án 'Smart City' work address Đà Nẵng)]                            
"""

# 定义任务提示
prompt = f"""
    Task: Extract entities and their relationships from a given text. Extract according to the given entity relationship type and output according to the output format
    Entity type: enterprise, person, location, project.
    Type of relationship: enterprise and enterprise: Partnership, litigation, investment, acquired, branch
    People and enterprise: legal representatives, administrative personnel, litigants, shareholders
    enterprise and location: registered address, branch address, work address.
    enterprise and project: belong, investment, participation.
    Output format: [entity type: entity, triplet:(entity, relationship, entity)]
    There are two examples of the input text and the output format:{examples}
    If there is no such entity relationship, null is output. The relationship between the extracted entities is identified. 
    Output strictly in accordance with the format and only the final result is required
    """

# 初始化计数器，用于跟踪进度
total_articles = len(json_data)  # 总文章数
processed_count = 0  # 已处理文章数

# 遍历 JSON 数据中的每篇文章
for i, article in enumerate(json_data):
    # 如果文章已经包含 'entity_relationship' 字段，跳过处理
    if "entity_relationship" in article:
        print(f"跳过文章 {article['news_id']}，已处理。")
        processed_count += 1
        continue

    content = article["content"]  # 提取文章内容
    news_id = article["news_id"]  # 提取文章 ID

    print(f"正在处理文章 {i + 1}/{total_articles}: {news_id}")

    try:
        # 调用 Deepseek API 处理文章内容
        response = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            stream=False
        )

        # 提取 API 返回的结果
        result = response.choices[0].message.content
        article["entity_relationship"] = result  # 将结果保存到文章中

        print(f"文章 ID: {news_id}")
        print(f"内容: {content}")
        print(f"实体关系: {result}")
        print("-" * 50)

        processed_count += 1

    except Exception as e:
        print(f"处理文章 {news_id} 时出错: {e}")  # 使用 print 输出错误信息
        article["entity_relationship"] = "处理文章时出错"
        processed_count += 1

    # 每处理 100 条数据保存一次进度
    if (i + 1) % 100 == 0:
        print(f"已处理 {i + 1} 篇文章。正在保存进度...")
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(json_data, file, ensure_ascii=False, indent=4)
            print(f"进度已保存到 '{output_file}'。")
        except Exception as e:
            print(f"保存进度到文件时出错: {e}")  # 使用 print 输出错误信息

# 处理完所有数据后，最终保存一次
try:
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)
    print(f"处理完成。所有结果已保存到 '{output_file}'。")
except Exception as e:
    print(f"保存最终结果到文件时出错: {e}")  # 使用 print 输出错误信息
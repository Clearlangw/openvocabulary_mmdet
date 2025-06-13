import os
from openai import OpenAI
import json
import time
# ====== 1. 配置您的信息 ======
# 请替换为您的 OpenAI API 密钥
API_KEY = "sk-a78xK2cAqiMgyAAPD99c229b4114476fB6A46aCa42E98e48"  

# 如果您使用中转服务，请填写对应的地址，否则留空字符串 ""
API_BASE_URL = "https://aihubmix.com/v1" 

# 指定要使用的模型名称
MODEL_NAME ="gpt-4.1"

# 生成响应的最大 Token 数量
MAX_TOKENS_API = 256

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van", 
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

# 每个类别要生成的父类数量 (p)
P_SUPER_CATEGORIES = 3

# 每个类别要生成的子类数量 (q)
Q_SUB_CATEGORIES = 10

# 为每个类别调用 LLM 的次数 (t)
T_LLM_QUERIES = 3

# 使用的上下文，根据文献设置为 'object'
CONTEXT = "object"

# =================================

def call_e2e_gpt(prompt_text):
    """
    使用文本 prompt 调用 OpenAI API 并返回结果。

    Args:
        prompt_text (str): 您想向 GPT提出的问题或指令。

    Returns:
        str: GPT模型生成的文本响应，或者在出错时返回 None。
    """
    print("正在初始化 OpenAI 客户端...")
    try:
        # 根据是否提供了 API_BASE_URL 来初始化客户端
        if API_BASE_URL:
            print(f"使用自定义 API 地址: {API_BASE_URL}")
            client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        else:
            print("使用默认 OpenAI API 地址。")
            client = OpenAI(api_key=API_KEY)
            
    except Exception as e:
        print(f"错误：初始化 OpenAI 客户端失败: {e}")
        return None

    print(f"正在使用模型 '{MODEL_NAME}' 发送您的请求...")
    try:
        # 发起 API 调用
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=MAX_TOKENS_API
        )
        # 返回模型生成的具体文本内容
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"错误：调用 OpenAI API 时出错: {e}")
        return None


def parse_llm_response(response_text):
    """
    一个更强大的解析函数，用于清理LLM可能返回的各种格式。
    - 优先处理 '&' 分隔符。
    - 如果失败，则处理换行符和编号列表。
    """
    if not response_text:
        return []

    # 方案 A: 理想情况，模型遵守了 '&' 分隔指令
    if '&' in response_text:
        # 即使有'&'，也可能混杂换行，先替换掉
        items = response_text.replace('\n', '&').split('&')
        # 清理并去除空字符串
        return [item.strip() for item in items if item.strip()]

    # 方案 B: 模型返回了编号或换行的列表（最常见的不合规情况）
    # 使用正则表达式移除行首的数字、点、星号、破折号和空格
    cleaned_items = []
    lines = response_text.split('\n')
    for line in lines:
        # re.sub(pattern, repl, string)
        # ^\s* : 匹配行首的任意空格
        # (\d+\.|-|\*) : 匹配 "1." 或 "-" 或 "*"
        # \s* : 匹配后面的任意空格
        cleaned_line = re.sub(r'^\s*(\d+\.|\-|\*)\s*', '', line.strip())
        if cleaned_line: # 确保清理后不是空字符串
            cleaned_items.append(cleaned_line)
            
    return cleaned_items

def call_gpt_api(prompt_text, client):
    """
    使用单个已初始化的客户端实例调用 OpenAI API。

    Args:
        prompt_text (str): 发送给 GPT 的完整指令。
        client (OpenAI): OpenAI 的客户端实例。

    Returns:
        str: GPT 模型生成的文本响应，或在出错时返回 None。
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=MAX_TOKENS_API,
            temperature=0.7 # 增加一点创造性，以获得多样化的答案
        )
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"    [!] 调用 API 时出错: {e}")
        return None

def generate_hierarchy_for_classes(class_list, client):
    """
    为列表中的每个类生成父类和子类层级。
    """
    final_hierarchy = {}

    for c in class_list:
        print(f"\n--- 正在处理类别: '{c}' ---")
        
        super_categories = set()
        sub_categories = set()

        # 1. 生成父类 (Super-categories) - 使用强化版Prompt
        print(f"  -> 正在生成父类 (执行 {T_LLM_QUERIES} 次查询)...")
        prompt_super = (
            f"Your task is to generate exactly {P_SUPER_CATEGORIES} super-categories for the object '{c}'.\n"
            f"The context is a general '{CONTEXT}'.\n"
            "Follow these rules STRICTLY:\n"
            "1. Output a list of the categories separated ONLY by the '&' symbol.\n"
            "2. DO NOT use numbers, bullets, or newlines.\n"
            "3. DO NOT add any extra explanation or text.\n"
            "Example format: category1&category2&category3\n\n"
            f"Generate the super-categories for: {c}"
        )
        for i in range(T_LLM_QUERIES):
            print(f"    查询 {i+1}/{T_LLM_QUERIES}...")
            api_response = call_gpt_api(prompt_super, client)
            if api_response:
                # 使用我们强大的新解析函数
                results = parse_llm_response(api_response)
                super_categories.update(results)
            time.sleep(1)

        # 2. 生成子类 (Sub-categories) - 使用强化版Prompt
        print(f"  -> 正在生成子类 (执行 {T_LLM_QUERIES} 次查询)...")
        prompt_sub = (
            f"Your task is to generate exactly {Q_SUB_CATEGORIES} types/sub-categories of the '{c}' object.\n"
            f"The context is a general '{CONTEXT}'.\n"
            "Follow these rules STRICTLY:\n"
            "1. Output a list of the types separated ONLY by the '&' symbol.\n"
            "2. DO NOT use numbers, bullets, or newlines.\n"
            "3. DO NOT add any extra explanation or text.\n"
            "Example format: type1&type2&type3&...&type10\n\n"
            f"Generate the sub-categories for: {c}"
        )
        for i in range(T_LLM_QUERIES):
            print(f"    查询 {i+1}/{T_LLM_QUERIES}...")
            api_response = call_gpt_api(prompt_sub, client)
            if api_response:
                # 同样使用新解析函数
                results = parse_llm_response(api_response)
                sub_categories.update(results)
            time.sleep(1)

        # 存储最终结果 (将集合转换为排序后的列表)
        final_hierarchy[c] = {
            "super-categories": sorted(list(super_categories)),
            "sub-categories": sorted(list(sub_categories))
        }

    return final_hierarchy

if __name__ == "__main__":
    # 检查 API_KEY 是否已设置
    if not API_KEY or "xxxxxxxxxxxxxxxxxxxxxx" in API_KEY:
        print("\n错误：请在脚本顶部的配置区域设置您的 'API_KEY'。")
    else:
        # ======================================
        print("正在初始化 OpenAI 客户端...")
        if API_BASE_URL:
            print(f"使用自定义 API 地址: {API_BASE_URL}")
            openai_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        else:
            print("使用默认 OpenAI API 地址。")
            openai_client = OpenAI(api_key=API_KEY)
        # 调用函数并获取响应
        print(f"模型: {MODEL_NAME} | 单类别查询次数: {T_LLM_QUERIES}")

        # 执行主任务
        hierarchy_results = generate_hierarchy_for_classes(CLASS_NAMES, openai_client)

        # 打印最终结果
        print("\n" + "="*50)
        print("✅ 任务完成！生成的层级关系如下:")
        print("="*50)

        # 使用 json 模块美化输出
        formatted_results = json.dumps(hierarchy_results, indent=4)
        print(formatted_results)

        # 将结果保存到文件
        output_filename = "class_hierarchy.json"
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(formatted_results)
            print(f"\n[成功] 结果已保存到文件: {output_filename}")
        except Exception as e:
            print(f"\n[错误] 保存文件时出错: {e}")

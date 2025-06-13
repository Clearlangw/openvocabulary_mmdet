# generate_class_attributes_complete.py
import os
import json
import time
import re  # 导入正则表达式库用于数据清洗
from openai import OpenAI

# ====== 1. 配置您的信息 ======
# 请替换为您的 OpenAI API 密钥
API_KEY = "sk-a78xK2cAqiMgyAAPD99c229b4114476fB6A46aCa42E98e48"  

# 如果您使用中转服务，请填写对应的地址，否则留空字符串 ""
API_BASE_URL = "https://aihubmix.com/v1" 

# 指定要使用的模型名称
MODEL_NAME ="gpt-4.1"

# 生成响应的最大 Token 数量
MAX_TOKENS_API = 256

# ====== 2. 任务配置 ======
# 需要生成层级关系的类别列表
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

# --- 父类/子类配置 ---
P_SUPER_CATEGORIES = 3
Q_SUB_CATEGORIES = 10
T_LLM_QUERIES = 3
CONTEXT = "object"

# --- 近似词配置 ---
# 最终需要为每个类别生成的近似词数量
NUM_SYNONYMS_TO_GENERATE = 3
# 为了有足够选择空间，向API请求更多候选词
NUM_SYNONYMS_TO_REQUEST = 8

# ============================================

def parse_llm_response(response_text):
    """
    一个更强大的解析函数，用于清理LLM可能返回的各种格式。
    - 优先处理 '&' 分隔符。
    - 如果失败，则处理换行符和编号列表。
    - 所有结果转为小写以方便比较和去重。
    """
    if not response_text:
        return []

    # 方案 A: 理想情况，模型遵守了 '&' 分隔指令
    if '&' in response_text:
        # 即使有'&'，也可能混杂换行，先替换掉
        items = response_text.replace('\n', '&').split('&')
        return [item.strip().lower() for item in items if item.strip()]

    # 方案 B: 模型返回了编号或换行的列表
    cleaned_items = []
    lines = response_text.split('\n')
    for line in lines:
        # 使用正则表达式移除行首的数字、点、星号、破折号和空格
        cleaned_line = re.sub(r'^\s*(\d+\.|\-|\*)\s*', '', line.strip())
        if cleaned_line:
            cleaned_items.append(cleaned_line.lower())

    return cleaned_items


def call_gpt_api(prompt_text, client, temperature=0.6):
    """使用单个已初始化的客户端实例调用 OpenAI API。"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=MAX_TOKENS_API,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"    [!] 调用 API 时出错: {e}")
        return None


def process_all_classes(class_list, client):
    """
    为所有类别生成父类、子类和唯一的近似词。
    """
    final_results = {}
    # 1. 初始化全局屏蔽列表，包含所有基础类别名称（小写）
    global_blocklist = set(c.lower() for c in class_list)

    for class_name in class_list:
        print(f"\n--- 正在处理类别: '{class_name}' ---")
        final_results[class_name] = {}
        super_categories = set()
        sub_categories = set()

        # 2. 生成父类 (Super-categories)
        print(f"  -> 正在生成父类 (执行 {T_LLM_QUERIES} 次查询)...")
        prompt_super = (
            f"Your task is to generate exactly {P_SUPER_CATEGORIES} super-categories for the object '{class_name}'.\n"
            "Follow these rules STRICTLY:\n"
            "1. Output a list of the categories separated ONLY by the '&' symbol.\n"
            "2. DO NOT use numbers, bullets, or newlines.\n"
            "3. DO NOT add any extra explanation or text.\n"
            "Example format: category1&category2&category3\n\n"
            f"Generate the super-categories for: {class_name}"
        )
        for i in range(T_LLM_QUERIES):
            print(f"    父类查询 {i+1}/{T_LLM_QUERIES}...")
            api_response = call_gpt_api(prompt_super, client, temperature=0.7)
            if api_response:
                results = parse_llm_response(api_response)
                super_categories.update(results)
            time.sleep(1) # API调用间的友好停顿
        final_results[class_name]["super-categories"] = sorted(list(super_categories))
        other_main_classes_str = ", ".join([c for c in CLASS_NAMES if c.lower() != class_name.lower()])
        # 3. 生成子类 (Sub-categories)
        print(f"  -> 正在生成子类 (执行 {T_LLM_QUERIES} 次查询)...")
        prompt_sub = (
            f"Your task is to generate exactly {Q_SUB_CATEGORIES} types/sub-categories of the '{class_name}' object.\n"
            f"CRITICAL RULE: The sub-categories must be specific types of '{class_name}'. They CANNOT be any of the other main categories, such as: {other_main_classes_str}.\n"
            " For example, if generating for 'pedestrian', avoid terms like 'adult' which could also describe a 'people'.\n"
            "Follow these rules STRICTLY:\n"
            "1. Output a list of the types separated ONLY by the '&' symbol.\n"
            "2. DO NOT use numbers, bullets, or newlines.\n"
            "3. DO NOT add any extra explanation or text.\n"
            "Example format: type1&type2&type3&...&type10\n\n"
            f"Generate the sub-categories for: {class_name}"
        )
        for i in range(T_LLM_QUERIES):
            print(f"    子类查询 {i+1}/{T_LLM_QUERIES}...")
            api_response = call_gpt_api(prompt_sub, client, temperature=0.7)
            if api_response:
                results = parse_llm_response(api_response)
                sub_categories.update(results)
            time.sleep(1)
        final_results[class_name]["sub-categories"] = sorted(list(sub_categories))
        
        # 4. 生成唯一的近似词 (Synonyms)
        # print(f"  -> 正在生成唯一的近似词...")
        # blocklist_str = ", ".join(sorted(list(global_blocklist)))
        # prompt_synonym = (
        #     f"Generate a list of {NUM_SYNONYMS_TO_REQUEST} common synonyms or approximate terms for the object '{class_name}'.\n"
        #     f"CRITICAL: DO NOT use any of the words from the following blocklist: {blocklist_str}\n"
        #     "Follow these rules STRICTLY:\n"
        #     "1. Output a list separated ONLY by the '&' symbol.\n"
        #     "2. DO NOT use numbers, bullets, or newlines.\n"
        #     "3. Rank the synonyms from most common to least common.\n\n"
        #     f"Generate synonyms for: {class_name}"
        # )
        # api_response_syn = call_gpt_api(prompt_synonym, client, temperature=0.5)
        print(f"  -> 正在生成唯一的近似词...")
        # 4a. 构造只包含其他核心类别的列表，用于语义约束
        other_main_classes_str = ", ".join([c for c in CLASS_NAMES if c.lower() != class_name.lower()])
        
        prompt_synonym = (
            f"Generate a list of {NUM_SYNONYMS_TO_REQUEST} common synonyms or approximate terms for the object '{class_name}'.\n"
            f"CRITICAL: The synonyms must be specific to '{class_name}'. They must NOT be broad terms or words that are conceptually similar to any other main object categories in this list: {other_main_classes_str}."
            " For example, if generating for 'truck', avoid terms like 'large vehicle' which could also describe a 'bus' or 'van'.\n"
            "Follow these rules STRICTLY:\n"
            "1. Output a list separated ONLY by the '&' symbol.\n"
            "2. DO NOT use numbers, bullets, or newlines.\n"
            "3. Rank the synonyms from most common to least common.\n\n"
            f"Generate synonyms for: {class_name}"
        )
        api_response_syn = call_gpt_api(prompt_synonym, client, temperature=0.5)
        
        valid_synonyms = []
        if api_response_syn:
            candidate_synonyms = parse_llm_response(api_response_syn)
            for synonym in candidate_synonyms:
                if synonym and synonym not in global_blocklist:
                    valid_synonyms.append(synonym)
                    if len(valid_synonyms) >= NUM_SYNONYMS_TO_GENERATE:
                        break # 已找到足够数量
        
        if len(valid_synonyms) < NUM_SYNONYMS_TO_GENERATE:
            print(f"    [!] 警告：只为 '{class_name}' 找到了 {len(valid_synonyms)}/{NUM_SYNONYMS_TO_GENERATE} 个唯一近似词。")
        
        final_results[class_name]["synonyms"] = sorted(valid_synonyms)

        # 5. **关键步骤**：将新生成的、有效的近似词更新到全局屏蔽列表
        print(f"    [+] 已为 '{class_name}' 找到近似词: {valid_synonyms}")
        global_blocklist.update(valid_synonyms)
        print(f"    [*] 当前屏蔽列表大小: {len(global_blocklist)}")

    return final_results


# 主执行代码块
if __name__ == "__main__":
    if not API_KEY or "xxxxxxxxxxxxxxxxxxxxxx" in API_KEY:
        print("\n[错误] 请在脚本顶部的配置区域设置您的 'API_KEY'。")
    else:
        print("正在初始化 OpenAI 客户端...")
        try:
            client_args = {"api_key": API_KEY}
            if API_BASE_URL:
                client_args["base_url"] = API_BASE_URL
                print(f"使用自定义 API 地址: {API_BASE_URL}")
            else:
                print("使用默认 OpenAI 地址。")

            openai_client = OpenAI(**client_args)

            print("任务配置:")
            print(f"  - 模型: {MODEL_NAME}")
            print(f"  - 目标类别: {len(CLASS_NAMES)}个")
            print(f"  - 父类/子类查询次数: {T_LLM_QUERIES}次/每项")
            print(f"  - 目标近似词数量: {NUM_SYNONYMS_TO_GENERATE}个/每项")

            # 执行主任务
            results = process_all_classes(CLASS_NAMES, openai_client)

            print("\n" + "="*60)
            print("✅ 任务完成！所有类别的完整属性已生成:")
            print("="*60)

            # 使用 json 模块美化输出并确保中文字符正常显示
            formatted_results = json.dumps(results, indent=4, ensure_ascii=False)
            print(formatted_results)

            output_filename = "class_hierarchy_complete.json"
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(formatted_results)
                print(f"\n[成功] 结果已完整保存到文件: {output_filename}")
            except Exception as e:
                print(f"\n[错误] 保存文件时出错: {e}")

        except Exception as e:
            print(f"\n[致命错误] 程序执行失败: {e}")
import json

# ====== 1. 配置 ======
# 输入的JSON文件名 (由之前的脚本生成)
INPUT_JSON_FILE = "class_hierarchy_complete.json"

# 输出的JSON文件名
OUTPUT_SENTENCES_JSON_FILE = "generated_sentences_by_class.json"

# 是否在句子的中间部分使用近似词 (synonyms)。
USE_SYNONYMS = True
# ========================

def load_data(filename):
    """从JSON文件加载数据并处理错误。"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            print(f"✅ 成功加载数据文件: {filename}")
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 输入文件未找到 -> {filename}")
        return None
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件 {filename} 不是有效的JSON格式。")
        return None

def get_article(word):
    """根据单词首字母返回正确的冠词 'a' 或 'an'。"""
    if not word:
        return "a"
    return "an" if word[0] in "aeiou" else "a"

def generate_sentences_by_class(data):
    """
    根据加载的数据，生成一个以主类别为键，句子列表为值的字典。
    """
    if not data:
        return {}

    # 最终的数据结构将是一个字典，而不是列表
    sentences_by_class = {}
    total_sentence_count = 0
    print("\n--- 开始生成句子 ---")

    # 遍历JSON中的每一个主类别 (e.g., "car", "truck")
    for main_class, attributes in data.items():
        # 为当前主类别初始化一个空列表来存放句子
        sentences_by_class[main_class] = []
        
        main_class_lower = main_class.lower()
        
        # 数据验证和准备
        sub_categories = attributes.get("sub-categories", [])
        super_categories = attributes.get("super-categories", [])
        synonyms = attributes.get("synonyms", [])

        if not sub_categories or not super_categories:
            print(f"🟡 警告: 类别 '{main_class}' 缺少子类或父类，已跳过。")
            continue

        # 准备句子的中间部分
        middle_terms = {main_class_lower}
        if USE_SYNONYMS and synonyms:
            middle_terms.update([s.lower() for s in synonyms])
        
        # 生成所有句子组合
        for sub_cat in sub_categories:
            sub_cat_lower = sub_cat.lower()
            article1 = get_article(sub_cat_lower)

            for middle in middle_terms:
                for super_cat in super_categories:
                    super_cat_lower = super_cat.lower()
                    
                    sentence = (
                        f"{article1} {sub_cat_lower}, "
                        f"which is a {middle}, "
                        f"which is a {super_cat_lower}"
                    )
                    # 将生成的句子添加到对应主类别的列表中
                    sentences_by_class[main_class].append(sentence)
        
        class_sentence_count = len(sentences_by_class[main_class])
        total_sentence_count += class_sentence_count
        print(f"  已为 '{main_class}' 生成 {class_sentence_count} 条句子。")

    print(f"\n总计生成 {total_sentence_count} 条句子。")
    return sentences_by_class

def save_to_json(data_dict, filename):
    """将字典数据以美化的格式保存到JSON文件。"""
    if not data_dict:
        print("🟡 警告: 没有生成任何数据，未创建文件。")
        return

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # 使用 json.dump 来写入数据
            # indent=4: 美化输出，自动缩进4个空格
            # ensure_ascii=False: 确保中文字符或其他非ASCII字符能被正确写入
            json.dump(data_dict, f, indent=4, ensure_ascii=False)
        print(f"\n✅ 成功! 数据已按类别保存到JSON文件: {filename}")
    except IOError as e:
        print(f"❌ 错误: 无法写入文件 {filename}。错误信息: {e}")

# 主执行函数
if __name__ == "__main__":
    # 步骤 1: 加载数据
    class_data = load_data(INPUT_JSON_FILE)

    if class_data:
        # 步骤 2: 生成按类别组织的句子字典
        sentences_data = generate_sentences_by_class(class_data)
        
        # 步骤 3: 将字典保存为JSON文件
        save_to_json(sentences_data, OUTPUT_SENTENCES_JSON_FILE)
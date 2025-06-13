# text_analyzer_script.py
import os
import argparse
import re # 正则表达式库
from collections import Counter # 计数器，用于词频统计
from pathlib import Path # 路径操作库
from wordcloud import WordCloud # 词云生成库 (需要安装: pip install wordcloud)
import matplotlib.pyplot as plt # 绘图库 (需要安装: pip install matplotlib)

# 默认英文停用词列表 - 更加通用和精简
# 主要包含语言结构词和非常通用的上下文无关词
DEFAULT_STOP_WORDS = set([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "can", "could", "may", "might", "must",
    "and", "but", "or", "nor", "for", "so", "yet", "in", "on", "at", "by", "from", "to", "with", "of",
    "about", "above", "after", "again", "against", "all", "am", "as", "any", "because", "before",
    "below", "between", "both", "during", "each", "few", "further", "here", "how", "i", "if", "into",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "not", "now", "once",
    "only", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "he", "him",
    "her", "herself", "himself", "some", "such", "than", "that", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "too", "under",
    "until", "up", "very", "we", "what", "when", "where", "which", "while", "who", "whom", "why",
    "you", "your", "yours", "yourself", "yourselves",
    "side","front","back","left","right","top","bottom","top-left","top-right","bottom-left","bottom-right",
    # 图像描述和API调用中非常常见的通用词，这些词通常不携带特定对象的描述信息
    "image", "images", "object", "objects", "referred", "visible", "feature", "features", "distinct", "important",
    "also", "primarily", "vehicle", "person", # "vehicle" 和 "person" 作为非常高阶的类别，如果不是分析目标本身，通常停用
    "key", "components", "component", "structure", "shape", # "structure" 和 "shape" 过于通用
    "objective", "physical", "attributes", "example", "series", "short", "factual", "phrases",
    "file", "text", "name", "image1", "image2", "task", "compare", "comparison", "differences", "difference",
    "following", "format", "strictly", "focus", "solely", "primary", "avoid", "assumptions",
    "interpretations", "imaginative", "descriptions", "present", "target", "type", "specific",
    "notable", "detail", "describe", "description", "detailed", "provide", "ignore", "color",
    "colors", "aclass", "bclass", "featuring", "vs", "sorry", "cant", "determine",
    "due", "low", "resolution", "quality", "unclear", "accurately", "contents", "blur", "blurriness",
    "lack", "clear", "details", "can't", "identify", "from"
    # 注意：具体的类别名（如 "awning-tricycle", "van"）已从此列表移除，将动态处理
])


def tokenize_and_clean_text(text_content, stop_words):
    """将文本转换为小写，移除标点符号，分词，并移除停用词。"""
    if not text_content: # 如果文本内容为空，返回空列表
        return []
    text_content = text_content.lower() # 转换为小写
    # 移除除字母、数字、空格、连字符外的所有字符 (连字符可能构成有效词的一部分，如 "three-wheeled")
    text_content = re.sub(r'[^\w\s-]', '', text_content) 
    words = text_content.split() # 分词
    # 过滤停用词和长度小于等于1的词（通常是单个字母或无意义的残留）
    return [word for word in words if word not in stop_words and len(word) > 1]

def get_word_frequencies(text_content, current_stop_words):
    """计算文本块中的词频，使用当前上下文的停用词列表。"""
    words = tokenize_and_clean_text(text_content, current_stop_words) # 获取清洗和分词后的词列表
    if not words: # 如果没有有效词汇，返回空的Counter对象
        return Counter()
    return Counter(words) # 返回包含词频的Counter对象

def save_word_frequencies_to_txt(word_counts, output_filepath, num_top_words_report=None):
    """将词频保存到文本文件。"""
    try:
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True) # 确保输出目录存在
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write("词频统计表:\n") # 文件头
            f.write("----------------------\n")
            f.write(f"{'词汇':<30} | {'频率'}\n") # 表头
            f.write("----------------------\n")
            
            items_to_write = word_counts.most_common(num_top_words_report) if num_top_words_report else word_counts.most_common()
            
            if not items_to_write: # 如果没有词汇可写
                f.write("过滤后未发现有效词汇。\n")
            else:
                for word, count in items_to_write: # 遍历并写入词汇和频率
                    f.write(f"{word:<30} | {count}\n")
        print(f"  已将词频表保存至: {output_filepath}")
    except Exception as e:
        print(f"错误：保存词频表至 {output_filepath} 时出错: {e}")

def generate_and_save_wordcloud(word_counts, output_filepath, current_stop_words, width=1200, height=600, background_color='white', colormap='viridis'):
    """根据词频生成词云图并保存。"""
    if not word_counts: # 如果词频为空，则跳过生成
        print(f"  由于没有可用词汇，跳过为 {output_filepath} 生成词云图。")
        return
    try:
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True) # 确保输出目录存在
        
        wc = WordCloud(width=width, height=height, background_color=background_color,
                       max_words=150, 
                       colormap=colormap, 
                       stopwords=current_stop_words, 
                       collocations=False, 
                       ).generate_from_frequencies(dict(word_counts))
        
        plt.figure(figsize=(width/100, height/100), dpi=100) 
        plt.imshow(wc, interpolation='bilinear') 
        plt.axis('off') 
        plt.tight_layout(pad=0) 
        plt.savefig(output_filepath) 
        plt.close() # 关键：关闭图像以释放内存，避免 `plt.show()`
        print(f"  已将词云图保存至: {output_filepath}")
    except Exception as e:
        print(f"错误：生成或保存词云图至 {output_filepath} 时出错: {e}")


def parse_aggregated_caption_file_content(filepath):
    """读取聚合的描述文件并返回其内容（移除非描述性标题行）。"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content_lines = []
            for line in f:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith("--- Caption for"):
                    content_lines.append(stripped_line)
            return " ".join(content_lines) 
    except FileNotFoundError:
        print(f"错误：聚合描述文件未找到: {filepath}")
        return ""
    except Exception as e:
        print(f"错误：读取聚合描述文件 {filepath} 时出错: {e}")
        return ""

def parse_aggregated_diff_file_content(filepath):
    """
    解析聚合的差异文件 (例如 folderA-folderB_diff.txt)。
    它收集分别属于对象A和对象B的所有特征描述。
    返回两个字符串: all_features_for_A, all_features_for_B。
    """
    all_features_for_A = []
    all_features_for_B = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误：聚合差异文件未找到: {filepath}")
        return "", ""
    except Exception as e:
        print(f"错误：读取聚合差异文件 {filepath} 时出错: {e}")
        return "", ""

    current_block_object_name = None 
    current_block_is_for_folder_A = False 

    for line in lines:
        line = line.strip()
        if not line: 
            continue

        obj_a_header_match = re.match(r"Object A Name \(extracted\):\s*(.*)", line)
        if obj_a_header_match:
            current_block_object_name = obj_a_header_match.group(1).strip()
            current_block_is_for_folder_A = True 
            continue 

        obj_b_header_match = re.match(r"Object B Name \(extracted\):\s*(.*)", line)
        if obj_b_header_match:
            current_block_object_name = obj_b_header_match.group(1).strip()
            current_block_is_for_folder_A = False 
            continue 
        
        if line.startswith("---"):
            continue

        if current_block_object_name:
            feature_match = re.match(rf"'{re.escape(current_block_object_name)}':\s*(.*)", line)
            if feature_match:
                features_text = feature_match.group(1).strip() 
                if current_block_is_for_folder_A:
                    all_features_for_A.append(features_text)
                else:
                    all_features_for_B.append(features_text)
                current_block_object_name = None 
    
    return " ".join(all_features_for_A), " ".join(all_features_for_B)


def analyze_text_files(
    caption_input_dir, diff_input_dir, analysis_output_dir, 
    num_top_words_report_individual, global_stop_words,
    num_top_words_report_overall, wordcloud_width_overall, wordcloud_height_overall
    ):
    """主函数，用于分析描述和差异文本文件，并进行整体分析。"""
    print("\n--- 开始文本分析 ---")
    overall_analysis_out_path = os.path.join(analysis_output_dir, "overall_analysis") # 整体分析的输出子目录
    Path(analysis_output_dir).mkdir(parents=True, exist_ok=True) 
    Path(overall_analysis_out_path).mkdir(parents=True, exist_ok=True)

    all_captions_text_combined = [] # 用于累积所有描述文本
    all_diff_features_A_combined = [] # 用于累积所有差异文本中对象A的特征
    all_diff_features_B_combined = [] # 用于累积所有差异文本中对象B的特征

    # 1. 分析聚合的描述文件 (并累积文本)
    caption_analysis_path = os.path.join(analysis_output_dir, "caption_analysis") 
    Path(caption_analysis_path).mkdir(parents=True, exist_ok=True)
    print(f"\n[1] 正在分析聚合描述文件，来源: {caption_input_dir}")
    if not os.path.isdir(caption_input_dir):
        print(f"  警告：描述文件输入目录 '{caption_input_dir}' 未找到。跳过描述分析。")
    else:
        for class_dir_name in os.listdir(caption_input_dir): 
            class_dir_path = os.path.join(caption_input_dir, class_dir_name)
            if os.path.isdir(class_dir_path):
                agg_caption_filename = f"{class_dir_name}_captions_all.txt"
                agg_caption_filepath = os.path.join(class_dir_path, agg_caption_filename)
                
                if os.path.isfile(agg_caption_filepath):
                    print(f"  正在处理类别 '{class_dir_name}' 的描述摘要: {agg_caption_filepath}")
                    content = parse_aggregated_caption_file_content(agg_caption_filepath)
                    if content:
                        all_captions_text_combined.append(content) # 累积文本

                        current_caption_stop_words = global_stop_words.copy()
                        current_caption_stop_words.add(class_dir_name.lower())
                        for part in class_dir_name.lower().split('-'):
                            if len(part) > 1 : 
                                current_caption_stop_words.add(part)
                        
                        word_counts = get_word_frequencies(content, current_caption_stop_words)
                        
                        class_analysis_out_path = os.path.join(caption_analysis_path, class_dir_name)
                        Path(class_analysis_out_path).mkdir(parents=True, exist_ok=True)

                        freq_txt_path = os.path.join(class_analysis_out_path, f"{class_dir_name}_caption_freq.txt")
                        save_word_frequencies_to_txt(word_counts, freq_txt_path, num_top_words_report_individual)
                        
                        cloud_png_path = os.path.join(class_analysis_out_path, f"{class_dir_name}_caption_cloud.png")
                        # 单个文件的词云图尺寸可以使用默认或单独参数，这里为了简化，复用overall的，也可单独加参数
                        generate_and_save_wordcloud(word_counts, cloud_png_path, current_caption_stop_words, width=wordcloud_width_overall, height=wordcloud_height_overall)
                    else:
                        print(f"  {agg_caption_filepath} 中无内容可分析。")
                else:
                    print(f"  警告：在 {class_dir_path} 中未找到聚合描述文件 '{agg_caption_filename}'。")

    # 2. 分析聚合的差异文件 (并累积文本)
    diff_analysis_path = os.path.join(analysis_output_dir, "difference_analysis") 
    Path(diff_analysis_path).mkdir(parents=True, exist_ok=True)
    print(f"\n[2] 正在分析聚合差异文件，来源: {diff_input_dir}")
    if not os.path.isdir(diff_input_dir):
        print(f"  警告：差异文件输入目录 '{diff_input_dir}' 未找到。跳过差异分析。")
    else:
        for diff_filename in os.listdir(diff_input_dir): 
            if diff_filename.endswith("_diff.txt") and '-' in diff_filename and 'obj' not in diff_filename:
                parts = diff_filename.replace("_diff.txt", "").split('-', 1) 
                if len(parts) == 2:
                    folder_A_name, folder_B_name = parts[0], parts[1]
                    agg_diff_filepath = os.path.join(diff_input_dir, diff_filename)
                    
                    print(f"  正在处理差异摘要: {agg_diff_filepath} (比较 '{folder_A_name}' 和 '{folder_B_name}')")
                    features_A_text, features_B_text = parse_aggregated_diff_file_content(agg_diff_filepath)

                    comparison_analysis_out_path = os.path.join(diff_analysis_path, f"{folder_A_name}_vs_{folder_B_name}")
                    Path(comparison_analysis_out_path).mkdir(parents=True, exist_ok=True)

                    if features_A_text:
                        all_diff_features_A_combined.append(features_A_text) # 累积文本

                        current_diff_A_stop_words = global_stop_words.copy()
                        current_diff_A_stop_words.add(folder_A_name.lower()) 
                        for part in folder_A_name.lower().split('-'):
                             if len(part) > 1 : current_diff_A_stop_words.add(part)
                        current_diff_A_stop_words.add(folder_B_name.lower()) 
                        for part in folder_B_name.lower().split('-'):
                             if len(part) > 1 : current_diff_A_stop_words.add(part)

                        word_counts_A = get_word_frequencies(features_A_text, current_diff_A_stop_words)
                        freq_A_txt_path = os.path.join(comparison_analysis_out_path, f"{folder_A_name}_features_freq.txt")
                        save_word_frequencies_to_txt(word_counts_A, freq_A_txt_path, num_top_words_report_individual)
                        cloud_A_png_path = os.path.join(comparison_analysis_out_path, f"{folder_A_name}_features_cloud.png")
                        generate_and_save_wordcloud(word_counts_A, cloud_A_png_path, current_diff_A_stop_words, width=wordcloud_width_overall, height=wordcloud_height_overall)
                    else:
                        print(f"    在 {diff_filename} 中未找到或提取出 '{folder_A_name}' 的区分性特征。")

                    if features_B_text:
                        all_diff_features_B_combined.append(features_B_text) # 累积文本

                        current_diff_B_stop_words = global_stop_words.copy()
                        current_diff_B_stop_words.add(folder_B_name.lower()) 
                        for part in folder_B_name.lower().split('-'):
                             if len(part) > 1 : current_diff_B_stop_words.add(part)
                        current_diff_B_stop_words.add(folder_A_name.lower()) 
                        for part in folder_A_name.lower().split('-'):
                             if len(part) > 1 : current_diff_B_stop_words.add(part)

                        word_counts_B = get_word_frequencies(features_B_text, current_diff_B_stop_words)
                        freq_B_txt_path = os.path.join(comparison_analysis_out_path, f"{folder_B_name}_features_freq.txt")
                        save_word_frequencies_to_txt(word_counts_B, freq_B_txt_path, num_top_words_report_individual)
                        cloud_B_png_path = os.path.join(comparison_analysis_out_path, f"{folder_B_name}_features_cloud.png")
                        generate_and_save_wordcloud(word_counts_B, cloud_B_png_path, current_diff_B_stop_words, width=wordcloud_width_overall, height=wordcloud_height_overall)
                    else:
                        print(f"    在 {diff_filename} 中未找到或提取出 '{folder_B_name}' 的区分性特征。")
                else:
                    print(f"  跳过文件 (差异摘要文件名格式不符合预期): {diff_filename}")
    
    # 3. 执行整体分析
    print("\n[3] 正在执行整体文本分析...")

    # 3.1 整体描述文本分析
    if all_captions_text_combined:
        print("  正在分析所有描述文本的整体词频...")
        combined_captions_str = " ".join(all_captions_text_combined)
        # 对于整体分析，我们使用全局停用词，不动态添加任何特定类别名
        overall_caption_word_counts = get_word_frequencies(combined_captions_str, global_stop_words)
        
        freq_overall_caption_txt_path = os.path.join(overall_analysis_out_path, "overall_captions_freq.txt")
        save_word_frequencies_to_txt(overall_caption_word_counts, freq_overall_caption_txt_path, num_top_words_report_overall)
        
        cloud_overall_caption_png_path = os.path.join(overall_analysis_out_path, "overall_captions_cloud.png")
        generate_and_save_wordcloud(overall_caption_word_counts, cloud_overall_caption_png_path, global_stop_words, 
                                    width=wordcloud_width_overall, height=wordcloud_height_overall)
    else:
        print("  没有累积的描述文本可进行整体分析。")

    # 3.2 整体差异文本分析
    all_diff_text_combined_str = " ".join(all_diff_features_A_combined + all_diff_features_B_combined)
    if all_diff_text_combined_str.strip(): # 检查合并后是否有内容
        print("  正在分析所有差异文本的整体词频...")
        overall_diff_word_counts = get_word_frequencies(all_diff_text_combined_str, global_stop_words)
        
        freq_overall_diff_txt_path = os.path.join(overall_analysis_out_path, "overall_differences_freq.txt")
        save_word_frequencies_to_txt(overall_diff_word_counts, freq_overall_diff_txt_path, num_top_words_report_overall)
        
        cloud_overall_diff_png_path = os.path.join(overall_analysis_out_path, "overall_differences_cloud.png")
        generate_and_save_wordcloud(overall_diff_word_counts, cloud_overall_diff_png_path, global_stop_words,
                                    width=wordcloud_width_overall, height=wordcloud_height_overall)
    else:
        print("  没有累积的差异文本可进行整体分析。")

    # 3.3 全体文本整体分析 (描述 + 差异)
    all_text_combined_str = combined_captions_str + " " + all_diff_text_combined_str if 'combined_captions_str' in locals() else all_diff_text_combined_str
    if all_text_combined_str.strip():
        print("  正在分析所有文本（描述+差异）的整体词频...")
        overall_all_text_word_counts = get_word_frequencies(all_text_combined_str, global_stop_words)

        freq_overall_all_txt_path = os.path.join(overall_analysis_out_path, "overall_all_text_freq.txt")
        save_word_frequencies_to_txt(overall_all_text_word_counts, freq_overall_all_txt_path, num_top_words_report_overall)

        cloud_overall_all_png_path = os.path.join(overall_analysis_out_path, "overall_all_text_cloud.png")
        generate_and_save_wordcloud(overall_all_text_word_counts, cloud_overall_all_png_path, global_stop_words,
                                    width=wordcloud_width_overall, height=wordcloud_height_overall)
    else:
        print("  没有累积的文本可进行全体分析。")


    print("\n--- 文本分析完成 ---")


def main():
    parser = argparse.ArgumentParser(
        description="分析生成的描述和差异文本文件，提取关键词并生成可视化结果，包括整体分析。"
    )
    parser.add_argument(
        "--caption_input_dir", 
        required=True, 
        help="包含前一个脚本生成的描述文件输出的目录 (例如 'image_captions_output')。"
    )
    parser.add_argument(
        "--diff_input_dir", 
        required=True, 
        help="包含前一个脚本生成的差异文件输出的目录 (例如 'image_differences_output')。"
    )
    parser.add_argument(
        "--analysis_output_dir", 
        default="text_analysis_results", 
        help="保存分析结果（词频表和词云图）的目录。"
    )
    parser.add_argument(
        "--num_top_words_report_individual", # 重命名以区分
        type=int, 
        default=20, 
        help="在单个文件词频统计报告中显示的最高频词汇数量。"
    )
    # 新增：整体分析的参数
    parser.add_argument(
        "--num_top_words_report_overall", 
        type=int, 
        default=50, # 整体分析可以显示更多词
        help="在整体分析词频统计报告中显示的最高频词汇数量。"
    )
    parser.add_argument(
        "--wordcloud_width_overall",
        type=int,
        default=1600, # 整体词云图可以更大
        help="整体分析生成的词云图宽度（像素）。"
    )
    parser.add_argument(
        "--wordcloud_height_overall",
        type=int,
        default=800, # 整体词云图可以更高
        help="整体分析生成的词云图高度（像素）。"
    )
    args = parser.parse_args()
    
    analyze_text_files(
        args.caption_input_dir,
        args.diff_input_dir,
        args.analysis_output_dir,
        args.num_top_words_report_individual, # 使用区分后的参数名
        DEFAULT_STOP_WORDS, 
        args.num_top_words_report_overall, # 新参数
        args.wordcloud_width_overall,      # 新参数
        args.wordcloud_height_overall     # 新参数
    )

if __name__ == "__main__":
    main()

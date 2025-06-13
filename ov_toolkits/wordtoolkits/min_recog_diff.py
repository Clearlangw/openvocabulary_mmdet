# minimized_test_script.py
import os
import base64
from openai import OpenAI # Uses the modern OpenAI library (v1.0+)
from pathlib import Path

# Global variable for API client
client = None

# ====== Configuration - 请在此处修改您的设置 ======
# API Configuration
API_KEY = "sk-a78xK2cAqiMgyAAPD99c229b4114476fB6A46aCa42E98e48"  # 您的 OpenAI API 密钥
API_BASE_URL = "https://aihubmix.com/v1"  # 您的中转 API 地址 (如果使用)
MODEL_NAME = "gpt-4.1"  # 使用的模型名称 (例如 "gpt-4-turbo", "gpt-4.1-2025-04-14")
MAX_TOKENS_API = 256  # 生成响应的最大 Token 数

# --- Task Configuration ---
# 选择要执行的任务: "caption" 或 "difference"
# "caption": 为单张图片生成描述
# "difference": 比较两张图片的差异
TASK_TO_RUN = "caption"  #  <--- 在这里选择 "caption" 或 "difference"
TASK_TO_RUN = "difference"
# --- Image Paths Configuration (根据 TASK_TO_RUN 修改) ---
# 如果 TASK_TO_RUN = "caption":
SINGLE_IMAGE_PATH_FOR_CAPTION = "/home/wuke_2024/datasets/VisDrone2019-Cropped/awning-tricycle/0000045_01500_d_0000086_obj1092_awning-tricycle.png"  # <--- 替换为您的单张图片路径

# 如果 TASK_TO_RUN = "difference":
IMAGE_A_PATH_FOR_DIFFERENCE = "/home/wuke_2024/datasets/VisDrone2019-Cropped/awning-tricycle/0000045_01500_d_0000086_obj1092_awning-tricycle.png"  # <--- 替换为您的第一张图片路径
IMAGE_B_PATH_FOR_DIFFERENCE = "/home/wuke_2024/datasets/VisDrone2019-Cropped/motor/9999942_00000_d_0000087_obj287_motor.png" # <--- 替换为您的第二张图片路径
# =====================================================

def get_mime_type(image_filename):
    """Determines the MIME type based on the image file extension."""
    ext = os.path.splitext(image_filename)[1].lower()
    if ext == '.png':
        return 'image/png'
    elif ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    return 'application/octet-stream'

def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"错误：图片文件未找到 {image_path}")
        return None
    except Exception as e:
        print(f"编码图片时出错 {image_path}: {e}")
        return None

def call_openai_vision_api(prompt_text, images_data, model_name, max_tokens_api):
    """Calls the OpenAI Vision API using the initialized client."""
    if not client:
        print("OpenAI 客户端未初始化。")
        return None

    messages_content = [{"type": "text", "text": prompt_text}]
    
    for img_data in images_data:
        base64_string = img_data.get("base64")
        mime_type = img_data.get("mime_type")

        if not base64_string or len(base64_string) < 100:
            print("警告：图片 base64 字符串可能无效或为空。")
            continue
        if not mime_type:
            print(f"警告：未提供图片的 MIME 类型。")
            mime_type = 'image/jpeg' 

        messages_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_string}"
            }
        })
    
    if len(messages_content) <= 1 and images_data:
        print("错误：编码后没有有效的图片发送到 API。")
        return None

    try:
        print(f"正在使用模型 '{model_name}' 调用 API...")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": messages_content}],
            max_tokens=max_tokens_api
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"使用模型 {model_name} 调用 OpenAI API 时出错: {e}")
        return None

def initialize_client():
    """Initializes the OpenAI API client."""
    global client
    try:
        if API_BASE_URL:
            print(f"使用自定义 API 基础 URL: {API_BASE_URL}")
            client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        else:
            print("使用默认 OpenAI API 基础 URL。")
            client = OpenAI(api_key=API_KEY)
        print(f"将使用模型: {MODEL_NAME}")
        print(f"API 最大 Token 数: {MAX_TOKENS_API}")
    except Exception as e:
        print(f"初始化 OpenAI 客户端失败: {e}")
        client = None # Ensure client is None if initialization fails

def run_caption_task():
    """Generates a caption for a single image."""
    print(f"\n--- 执行描述生成任务 ---")
    if not os.path.exists(SINGLE_IMAGE_PATH_FOR_CAPTION):
        print(f"错误：找不到用于描述的图片：{SINGLE_IMAGE_PATH_FOR_CAPTION}")
        return

    print(f"正在为图片生成描述: {SINGLE_IMAGE_PATH_FOR_CAPTION}")
    image_base64 = encode_image_to_base64(SINGLE_IMAGE_PATH_FOR_CAPTION)
    if not image_base64:
        return

    image_name = os.path.basename(SINGLE_IMAGE_PATH_FOR_CAPTION)
    mime_type = get_mime_type(image_name)
    image_api_data = [{"base64": image_base64, "mime_type": mime_type}]

    # prompt = "Please provide a detailed description of any vehicles or people in this image."
    prompt = ("Focus solely on the primary vehicle or person in this image. "
              "Describe its objective physical attributes such as shape, structure, and key components. "
              "Avoid any assumptions, interpretations, or imaginative descriptions. "
              "Present the description as a series of short, factual phrases, for example: "
              "'A [target object type] with [specific shape/structure], featuring [component A], [component B], [notable detail C], [notable detail D].'")
    
    caption = call_openai_vision_api(prompt, image_api_data, MODEL_NAME, MAX_TOKENS_API)

    if caption:
        print("\n--- 生成的描述 ---")
        print(caption)
        # 您可以在此处添加代码将描述保存到文件
        # caption_file_name_base = os.path.splitext(image_name)[0]
        # output_path = f"./{caption_file_name_base}_caption_test.txt"
        # try:
        #     with open(output_path, 'w', encoding='utf-8') as f:
        #         f.write(caption)
        #     print(f"\n描述已保存到: {output_path}")
        # except Exception as e:
        #     print(f"保存描述时出错: {e}")
    else:
        print("未能生成描述。")

def extract_object_name_from_filename(image_filename_full):
    """Extracts a simplified object name from the image filename."""
    filename_base = os.path.splitext(os.path.basename(image_filename_full))[0]
    # Heuristic: take the last part after splitting by underscore.
    # This matches user examples like '..._obj1092_awning-tricycle.png' -> 'awning-tricycle'
    # or '..._obj287_motor.png' -> 'motor'
    # If no underscore, use the whole filename base.
    parts = filename_base.split('_')
    if len(parts) > 1:
        # Check if a part looks like objXXXX and take what's after if possible
        for i, part in enumerate(parts):
            if part.startswith("obj") and part[3:].isdigit():
                if i + 1 < len(parts):
                    return "-".join(parts[i+1:]) # Join remaining parts if name has hyphens
        return parts[-1] # Fallback to last part
    return filename_base # Fallback if no underscores or specific pattern

def run_difference_task():
    """Generates differences between two images."""
    print(f"\n--- 执行差异比较任务 ---")
    if not os.path.exists(IMAGE_A_PATH_FOR_DIFFERENCE) or not os.path.exists(IMAGE_B_PATH_FOR_DIFFERENCE):
        print(f"错误：找不到用于比较的一张或两张图片：")
        if not os.path.exists(IMAGE_A_PATH_FOR_DIFFERENCE):
            print(f"  - {IMAGE_A_PATH_FOR_DIFFERENCE}")
        if not os.path.exists(IMAGE_B_PATH_FOR_DIFFERENCE):
            print(f"  - {IMAGE_B_PATH_FOR_DIFFERENCE}")
        return

    print(f"正在比较图片 A: {IMAGE_A_PATH_FOR_DIFFERENCE}")
    print(f"和图片 B: {IMAGE_B_PATH_FOR_DIFFERENCE}")

    image_A_base64 = encode_image_to_base64(IMAGE_A_PATH_FOR_DIFFERENCE)
    image_B_base64 = encode_image_to_base64(IMAGE_B_PATH_FOR_DIFFERENCE)

    if not image_A_base64 or not image_B_base64:
        print("未能编码一张或两张图片以进行比较。")
        return
    
    # Extract object names from filenames
    object_A_name = extract_object_name_from_filename(IMAGE_A_PATH_FOR_DIFFERENCE)
    object_B_name = extract_object_name_from_filename(IMAGE_B_PATH_FOR_DIFFERENCE)
    
    print(f"  提取的目标名称 A: {object_A_name}")
    print(f"  提取的目标名称 B: {object_B_name}")

    mime_type_A = get_mime_type(os.path.basename(IMAGE_A_PATH_FOR_DIFFERENCE))
    mime_type_B = get_mime_type(os.path.basename(IMAGE_B_PATH_FOR_DIFFERENCE))

    images_api_data = [
        {"base64": image_A_base64, "mime_type": mime_type_A},
        {"base64": image_B_base64, "mime_type": mime_type_B}
    ]

    # Updated prompt for differences, incorporating object names
    prompt = (
        f"Image 1 is an image of an object referred to as '{object_A_name}'. "
        f"Image 2 is an image of an object referred to as '{object_B_name}'.\n\n"
        f"Your task is to compare '{object_A_name}' (visible in Image 1) and '{object_B_name}' (visible in Image 2).\n"
        "Focus on their distinct objective physical attributes such as shape, structure, and key components. Ignore color differences.\n\n"
        "Present the differences as a series of short, factual phrases for each object, strictly following this format:\n"
        f"'{object_A_name}': feature A1, feature A2, feature A3 (or more features if distinct and important).\n"
        f"'{object_B_name}': feature B1, feature B2, feature B3 (or more features if distinct and important)."
    )
    
    differences_text = call_openai_vision_api(prompt, images_api_data, MODEL_NAME, MAX_TOKENS_API)

    if differences_text:
        print("\n--- 生成的差异描述 ---")
        print(differences_text)
        # 您可以在此处添加代码将差异描述保存到文件
        # diff_file_name = f"./{object_A_name}_vs_{object_B_name}_diff_test.txt"
        # try:
        #     with open(diff_file_name, 'w', encoding='utf-8') as f:
        #         f.write(differences_text)
        #     print(f"\n差异描述已保存到: {diff_file_name}")
        # except Exception as e:
        #     print(f"保存差异描述时出错: {e}")
    else:
        print("未能生成差异描述。")

if __name__ == "__main__":
    if not API_KEY or "xxxxxxxxxxxxxxxxxxxxxx" in API_KEY:
        print("错误：请在脚本顶部的 'Configuration' 部分设置您的 API_KEY。")
    else:
        initialize_client()
        if client: # Proceed only if client was initialized successfully
            if TASK_TO_RUN == "caption":
                run_caption_task()
            elif TASK_TO_RUN == "difference":
                run_difference_task()
            else:
                print(f"错误：无效的 TASK_TO_RUN 值 '{TASK_TO_RUN}'。请选择 'caption' 或 'difference'。")
        else:
            print("由于客户端初始化失败，无法执行任务。")

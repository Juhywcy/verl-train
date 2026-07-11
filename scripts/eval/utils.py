import os

def get_json_file_paths(directory):
    """
    获取指定目录下所有 JSON 文件的路径。
    
    Args:
        directory (str): 要搜索的目录路径。
    
    Returns:
        list: 包含所有 JSON 文件路径的列表。
    """
    json_file_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_file_path = os.path.join(root, file)
                json_file_paths.append(json_file_path)
    
    return json_file_paths

def get_filtered_json_files(directory, filename_pattern):
    """
    获取指定目录下所有符合文件名规则的 JSON 文件路径。
    
    Args:
        directory (str): 要搜索的目录路径。
        filename_pattern (str): 文件名的匹配模式,可以包含通配符。
    
    Returns:
        list: 包含符合条件的 JSON 文件路径的列表。
    """
    json_files = []
    
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.jsonl') and f.startswith(filename_pattern):
                json_file_path = os.path.join(root, f)
                json_files.append(json_file_path)
    
    return json_files

if __name__ == "__main__":
    jsonl_file_path = "/home/hrzhu/code/long2short/ckpts/DAPO/DAPO-deepseek-Qwen1.5B/global_step_200/actor_hf/eval"
    json_files = get_filtered_json_files(jsonl_file_path, "samples_aime")
    print(json_files)
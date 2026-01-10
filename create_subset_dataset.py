import json
import os

def create_subset_dataset(input_path, output_path, max_size_mb):
    """
    从一个大的JSON文件中提取一个子集，并确保其为有效的JSON格式。

    Args:
        input_path (str): 输入的JSON文件路径。
        output_path (str): 输出的JSON文件路径。
        max_size_mb (int): 子集的最大大小（以MB为单位）。
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        # 写入JSON数组的开头
        outfile.write('[\n')
        
        # 跳过输入的第一个 '['
        infile.seek(1)
        
        buffer = ""
        first_object = True
        
        # 使用 ijson 逐个解析对象以节省内存
        # 由于我们没有 ijson，我们将手动解析
        # 这是一个简化的解析器，假设每个对象都在 '{' 和 '}' 之间
        
        brace_level = 0
        in_string = False
        current_object = ""
        
        while True:
            char = infile.read(1)
            if not char:
                print("Reached end of file.")
                continue

            current_object += char
            
            if char == '"':
                # 简单的字符串切换，不处理转义的引号
                in_string = not in_string
            
            if not in_string:
                if char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
                    
                    if brace_level == 0 and len(current_object.strip()) > 0:
                        # 找到了一个完整的对象
                        if not first_object:
                            outfile.write(',\n')
                        
                        # 去除前导的非 { 字符（如逗号和换行符）
                        obj_start_index = current_object.find('{')
                        if obj_start_index != -1:
                            clean_object = current_object[obj_start_index:]
                            outfile.write(clean_object)
                            first_object = False
                        
                        current_object = ""

                        # 检查文件大小
                        if outfile.tell() > max_size_bytes:
                            print(f"已达到 {max_size_mb}MB 的大小限制。")
                            break
        
        # 写入JSON数组的结尾
        outfile.write('\n]')
        print(f"成功创建子集数据集: {output_path}")


if __name__ == "__main__":
    input_file = '/root/llama-7b/datasets/wikipedia_en_1gb.json'
    output_file = '/root/llama-7b/datasets/wikipedia_en_300mb.json'
    subset_size_mb = 100
    
    create_subset_dataset(input_file, output_file, subset_size_mb)

import os
import re
from typing import List
from model.models import File, Function

def create_directory_summary(root_path):
    Files_summary = []
    num = 0
    # 提取项目根目录名（路径的最后一部分，如 blog-api）
    project_root_name = os.path.basename(os.path.normpath(root_path))
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(('.java', '.py')):
                afile_path = os.path.join(root, file)
                relative_path = os.path.relpath(afile_path, root_path)  # 转换为相对路径
                # 将相对路径转换为点分隔格式，并添加项目根目录名前缀
                file_path_parts = relative_path.replace("\\", "/").split("/")
                file_path_parts[-1] = file_path_parts[-1].replace(".java", "").replace(".py", "")
                file_path = f"{project_root_name}." + ".".join(file_path_parts)
                file_name = file_path.split(".")[-1]
                with open(afile_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                
                file_summary = File(
                    file_id=num,
                    file_name=file_name,
                    file_path=file_path,
                    file_code=file_content,
                    file_desc=file_name,
                    func_list=[],
                    file_txt_vector=[],
                    file_discode=""
                )
                print(file_summary.file_path+" "+file_summary.file_desc)
                Files_summary.append(file_summary)
                num += 1
    return Files_summary

def add_functions_to_files(files: List[File], functions: List[Function], language:str="python"):
    for function in functions:
        if language == "python":
            # 检查文件代码中是否包含函数代码
            for file in files:
                if function.func_code in file.file_code:
                    file.func_list.append(function)
                    break
        elif language == "java":
            class_name = function.func_file
            for file in files:
                if file.file_name == class_name:
                    file.func_list.append(function)
                    break


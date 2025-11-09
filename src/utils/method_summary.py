"""
这个模块用于生成方法/函数描述
"""
import pandas as pd
import os
import numpy as np
import json
import time
import random
import re
import ast
import tokenize
from io import StringIO
from typing import List
from pydantic import BaseModel, ValidationError
from openai import OpenAI
from transformers import RobertaTokenizer, T5ForConditionalGeneration

from model.models import Function
from utils.function_clustering import set_func_adj_matrix

# 从环境变量加载API配置
import os
from dotenv import load_dotenv

load_dotenv()

# 延迟初始化客户端，避免在导入时立即报错
client = None

def get_client():
    """获取OpenAI客户端，如果未初始化则初始化"""
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 环境变量未设置。"
                "请创建 .env 文件并设置 OPENAI_API_KEY 和 OPENAI_BASE_URL，"
                "或者设置环境变量。"
            )
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    return client

# 生成总结函数的提示词
Func_summary_template = """\nYou are a software engineer who is reverse engineering the code in a system to extract its design requirements and functional descriptions. 
code is below:
{code}
A. Project context positioning
- Module Attribution analysis:
{folder_structure}
- Dependency graph:
{dependence}
B. Function functionality destructuring
- Input/output parameter analysis: List and explain the data structure and purpose of all inputs and outputs
- Core logic flowchart: flowchart describing key processing steps in natural language
- Exception handling mechanism: error handling logic and boundary conditions are identified and illustrated
- If you can figure out the action object of the function, specify it; if you can't, leave it out
# Task:
-Give a description of the function based on the AB step, and extrapolate the functional requirements from the code implementation
-Non-functional requirements identification: infer non-functional requirements such as performance and security as reflected by code constraints
-Answer exactly as the code says. Don't introduce extra information
-Use the following format to answer:
Please return the response in the following JSON format:
{{
    "func_desc": "Function description",
    "func_flow": "Function flow",
    "func_notf": "Non-functional requirements",
}}
"""
class func_response(BaseModel):
    func_desc: str
    func_flow: str
    func_notf: str
    class Config:
        extra = "forbid"  # 严格禁止额外字段

def extract_comments_from_code(code: str, language: str="python") -> str:
    """
    从代码中提取注释
    支持Python和Java的常见注释格式
    
    Args:
        code: 函数代码字符串
        language: 代码语言，支持"python"或"java"
        
    Returns:
        提取的注释文本，多个注释用换行符连接
    """
    # 类型检查和转换：确保code是字符串类型
    if code is None:
        return ""
    if not isinstance(code, str):
        # 尝试转换为字符串
        try:
            code = str(code)
        except Exception:
            return ""
    
    # 如果转换后是空字符串或只包含空白字符
    if not code or not code.strip():
        return ""
    
    comments = []
    
    if language == "python":
        # 1. 提取Python docstring（使用ast）
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        comments.append(docstring)
        except (SyntaxError, ValueError):
            # 如果代码不完整无法解析，继续使用其他方法
            pass
        
        # 2. 提取Python单行注释 (# comment)
        try:
            tokens = tokenize.generate_tokens(StringIO(code).readline)
            for tok in tokens:
                if tok.type == tokenize.COMMENT:
                    # 移除注释符号(#)和前后空白
                    comment_text = tok.string.strip().lstrip('#').strip()
                    if comment_text and comment_text not in comments:
                        comments.append(comment_text)
        except (tokenize.TokenError, SyntaxError):
            # 如果tokenize失败，使用正则表达式作为备用
            single_line_comments = re.findall(r'#\s*(.+?)(?=\n|$)', code)
            for comment in single_line_comments:
                comment = comment.strip()
                if comment and comment not in comments:
                    comments.append(comment)
                
    elif language == "java":
        # 1. 提取JavaDoc注释 /** ... */
        javadoc_comments = re.findall(r'/\*\*\s*(.+?)\s*\*/', code, re.DOTALL)
        for comment in javadoc_comments:
            # 清理JavaDoc注释，移除每行的*符号和前导空格
            lines = []
            for line in comment.split('\n'):
                cleaned_line = line.strip().lstrip('*').strip()
                if cleaned_line:
                    lines.append(cleaned_line)
            cleaned_comment = '.'.join(lines).strip()
            if cleaned_comment and cleaned_comment not in comments:
                comments.append(cleaned_comment)
        
        # 2. 提取多行注释 /* ... */
        multiline_comments = re.findall(r'/\*\s*(.+?)\s*\*/', code, re.DOTALL)
        for comment in multiline_comments:
            # 清理多行注释，移除每行的*符号
            lines = []
            for line in comment.split('\n'):
                cleaned_line = line.strip().lstrip('*').strip()
                if cleaned_line:
                    lines.append(cleaned_line)
            cleaned_comment = '.'.join(lines).strip()
            if cleaned_comment and cleaned_comment not in comments:
                comments.append(cleaned_comment)
        
        # 3. 提取单行注释 // comment
        java_single_comments = re.findall(r'//\s*(.+?)(?=\n|$)', code)
        for comment in java_single_comments:
            comment = comment.strip()
            if comment and comment not in comments:
                comments.append(comment)
    
    # 合并所有注释为字符串
    if comments:
        return '.'.join(comments)
    else:
        return ""



def generate_function_description_by_comment(functions: List[Function], language:str="python") -> List[Function]:
    """
    从函数代码中提取注释作为函数描述
    
    Args:
        functions: 函数列表
        language: 代码语言，支持"python"或"java"
        
    Returns:
        更新后的函数列表，func_desc字段包含提取的注释
    """
    for function in functions:
        function.func_desc = extract_comments_from_code(function.func_code, language=language)
        print(f"Function ID: {function.func_id}, Name: {function.func_name}, Description: {function.func_desc}")
    return functions


def generate_function_descriptions(functions:List[Function], modelname:str="deepseek-v3", method_adj_matrix: np.ndarray=None, language:str="python") -> List[Function]:
    
    for function in functions:
        function.func_desc = extract_comments_from_code(function.func_code, language=language)
        if function.func_desc != "":
            continue
        row = method_adj_matrix[function.func_id]
        dependence_parts = []
        for j in np.nonzero(row)[0]:
            if j == function.func_id:
                continue
            dependence_parts.append(
                functions[j].func_fullName + "\n" + functions[j].func_code + "\n"
            )
        dependence = "".join(dependence_parts)
        prompt = Func_summary_template.format(
            code=function.func_code,
            folder_structure=function.func_fullName,
            dependence=dependence
        )
        try:
            response = call_with_retry(lambda: get_client().chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=modelname,
                response_format={"type": "json_object"},
                temperature=0.3, 
                top_p=0.95, 
                frequency_penalty=0.5, 
                presence_penalty=0.2 
            ))
            json_str = response.choices[0].message.content
            json_str = json_str.replace("```json", "").replace("```", "")
            result = func_response.model_validate(json.loads(json_str)) 
            function.func_desc = result.func_desc
            function.func_flow = result.func_flow
            function.func_notf = result.func_notf
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            function.func_desc = function.func_name.split(".")[-1]
            function.func_flow = function.func_code
            function.func_notf = ""
        except ValidationError as e:
            print(f"Pydantic验证失败: {e}")
            function.func_desc = function.func_name.split(".")[-1]
            function.func_flow = function.func_code
            function.func_notf = ""
        except Exception as e:
            print(f"其他错误: {e}")
            function.func_desc = function.func_name.split(".")[-1]
            function.func_flow = function.func_code
            function.func_notf = ""
        # 打印函数描述
        print(f"Function ID: {function.func_id}, Name: {function.func_name}")
        print(f"Description: {function.func_desc}")
        print(f"Flow: {function.func_flow}")
        print(f"Non-functional requirements: {function.func_notf}")

def call_with_retry(fn, retries=5, base_delay=0.5, max_delay=8.0):
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "429" not in msg and "RateLimit" not in msg and "upstream" not in msg:
                raise
            delay = min(max_delay, base_delay * (2 ** i)) * (1 + random.random() * 0.25)
            time.sleep(delay)
    return fn()

def function_name_summary(functions: List[Function]) -> List[Function]:
    for function in functions:
        function.func_desc = function.func_fullName
    return functions

def code_t5_summary(functions: List[Function], language:str="python") -> List[Function]:
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')
    
    for function in functions:
        function.func_desc = extract_comments_from_code(function.func_code, language=language)
        if function.func_desc != "":
            continue
        # 确保func_code是字符串类型
        text = function.func_code
        if not isinstance(text, str):
            if text is None:
                text = ""
            else:
                text = str(text)
        if not text or not text.strip():
            function.func_desc = function.func_name
            continue
        # 如果text超长则截断
        # CodeT5模型的最大输入长度通常是512 tokens
        max_length = 512
        # 先检查原始文本长度（不添加special tokens，更准确）
        temp_encoded = tokenizer(text, add_special_tokens=False)
        original_length = len(temp_encoded['input_ids'])
        
        # 使用truncation参数自动截断超长文本
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encoded.input_ids
        
        # 如果原始文本被截断，给出警告
        if original_length > max_length:
            print(f"警告: 函数 {function.func_name} 的代码过长 ({original_length} tokens)，已截断至 {max_length} tokens")

        generated_ids = model.generate(input_ids, max_length=40)
        function.func_desc = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return functions

def method_summary(output_dir: str, strategy: str, language:str="python") -> List[Function]:
    method_df = pd.read_csv(os.path.join(output_dir, "methods.csv"))
    functions = []

    for index, row in method_df.iterrows():
        function_fullName = row["method_signature"]
        function_name = function_fullName.split("(")[0].split(".")[-1]
        # 提取类/模块名（可能是最后第二段或最后一段）
        parts = function_fullName.split("(")[0].split(".")
        func_file = parts[-2] if len(parts) >= 2 else parts[-1]
        
        # 确保method_code是字符串类型
        method_code = row["method_code"]
        if pd.isna(method_code):
            method_code = ""
        elif not isinstance(method_code, str):
            method_code = str(method_code)
        
        function = Function(
            func_id=row["ID"],
            func_name=function_name,
            func_desc="",
            func_file=func_file,
            func_flow="",
            func_notf="",
            func_code=method_code,
            func_fullName=function_fullName,
            func_txt_vector=[]
        )
        functions.append(function)
    print(f"Total functions: {len(functions)}")
    # 加载邻接矩阵
    # 读取 CSV 并转换为 numpy 数组，确保是整数类型
    try:
        func_adj_matrix_df = pd.read_csv(os.path.join(output_dir, 'method_adj_matrix.csv'), header=None)
        # 转换为整数类型，处理可能的 NaN 或字符串
        func_adj_matrix_df = func_adj_matrix_df.fillna(0).astype(int)
        func_adj_matrix = func_adj_matrix_df.to_numpy()
        
        # 验证矩阵大小是否与函数数量匹配
        expected_size = len(functions)
        if func_adj_matrix.shape[0] != expected_size or func_adj_matrix.shape[1] != expected_size:
            print(f"警告: 矩阵大小 ({func_adj_matrix.shape}) 与函数数量 ({expected_size}) 不匹配，将调整矩阵大小")
            # 如果矩阵太大，截断；如果太小，用零填充
            if func_adj_matrix.shape[0] > expected_size:
                func_adj_matrix = func_adj_matrix[:expected_size, :expected_size]
            elif func_adj_matrix.shape[0] < expected_size:
                # 创建新的矩阵，用零填充
                new_matrix = np.zeros((expected_size, expected_size), dtype=int)
                new_matrix[:func_adj_matrix.shape[0], :func_adj_matrix.shape[1]] = func_adj_matrix
                func_adj_matrix = new_matrix
        
        set_func_adj_matrix(func_adj_matrix)
    except Exception as e:
        print(f"警告: 加载 method_adj_matrix.csv 失败: {e}")
        # 如果加载失败，创建一个空的零矩阵
        func_adj_matrix = np.zeros((len(functions), len(functions)), dtype=int)
        set_func_adj_matrix(func_adj_matrix)

    # 在生成函数描述之前，先检查现在的函数代码中是否有函数描述
    for function in functions:
        if function.func_desc != "":
            continue
        else:
            function.func_desc = extract_comments_from_code(function.func_code, language=language)
            if function.func_desc != "":
                continue
            else:
                function.func_desc = function.func_name

    if strategy == "function_name":
        return function_name_summary(functions)
    elif strategy == "code_t5":
        return code_t5_summary(functions, language=language)
    elif strategy == "llm":
        return generate_function_descriptions(functions, modelname=modelname, method_adj_matrix=method_adj_matrix, language=language)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
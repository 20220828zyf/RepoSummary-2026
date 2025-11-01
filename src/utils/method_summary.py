"""
这个模块用于生成方法/函数描述
"""
import pandas as pd
import os
import numpy as np
import json
import time
import random
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
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

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

def generate_function_descriptions(functions:List[Function], modelname:str="deepseek-v3", func_adj_matrix: np.ndarray=None) -> List[Function]:
    for function in functions:
        row = func_adj_matrix[function.func_id]
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
            response = call_with_retry(lambda: client.chat.completions.create(
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


def function_name_summary(output_dir: str, functions: List[Function]) -> List[Function]:
    for function in functions:
        function.func_desc = function.func_name
    return functions

def code_t5_summary(output_dir: str, functions: List[Function]) -> List[Function]:
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')
    
    for function in functions:
        text = function.func_code
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        generated_ids = model.generate(input_ids, max_length=40)
        function.func_desc = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Function ID: {function.func_id}, Name: {function.func_name}, Description: {function.func_desc}")
    return functions

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

def method_summary(output_dir: str, strategy: str) -> List[Function]:
    method_df = pd.read_csv(os.path.join(output_dir, "method.csv"))
    functions = []

    for index, row in method_df.iterrows():
        function_fullName = row["method_signature"]
        function_name = function_fullName.split("(")[0].split(".")[-1]
        # 提取类/模块名（可能是最后第二段或最后一段）
        parts = function_fullName.split("(")[0].split(".")
        func_file = parts[-2] if len(parts) >= 2 else parts[-1]
        
        function = Function(
            func_id=row["ID"]-1,
            func_name=function_name,
            func_desc="",
            func_file=func_file,
            func_flow="",
            func_notf="",
            func_code=row["method_code"],
            func_fullName=function_fullName,
            func_txt_vector=[]
        )
        functions.append(function)

    # 加载邻接矩阵
    func_adj_matrix_df = pd.read_csv(os.path.join(output_dir, 'method_adj_matrix.csv'), header=None).to_numpy()
    func_adj_matrix = func_adj_matrix_df[1:, 1:]
    set_func_adj_matrix(func_adj_matrix)

    if strategy == "function_name":
        return function_name_summary(output_dir, functions)
    elif strategy == "code_t5":
        return code_t5_summary(output_dir, functions)
    elif strategy == "llm":
        return generate_function_descriptions(functions, modelname, func_adj_matrix)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
import time
import json
import random
from pydantic import BaseModel, ValidationError
from typing import Any, Dict, List
from openai import OpenAI
from model.models import Feature, method_Cluster
import pandas as pd

# 从环境变量加载API配置
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

userstory_prompt = """
You are an engineer working on a software system and your goal is to reverse engineer User Storys from codes. You are given a list of ids and descriptions of the codes in the system below. 
# Code:
{code_content}
# Task:
A) Your first task is to construct 1-2 well written paragraphs that will guide your work in the next task.
In the paragraph, identify what is the core goal or user need addressed by all or most of the Code as well as the action, system behavior, or information that is provided to the user by the Code to support this core goal. Discuss how these actions relate to one another. Provide specific details that highlight how the Code provide the user the ability to perform each of these actions, focusing on the details that are most important for the scope and purposes of a User Story. Be as specific as possible.
Importantly, do not make an information up or make assumptions. Only use information directly from the Code.
B) Then use the actions you identified to create User Story When creating the User Story ensure that:
- Focused on the core goals and user needs identified above.
- Grouping related or overlapping actions across Code together.
- The User Story fits the following description:
* A User story is a concise, informal description of a software feature or functionality, written from the perspective of the end user.    It is a key artifact in Agile software development, used to capture and prioritize requirements.    User stories differ from other software artifacts, such as use cases or functional specifications, in that they are intentionally brief, focusing on the user's needs and goals, and are typically written in everyday language. They are meant to foster collaboration and communication between the development team and stakeholders, rather than serving as a detailed technical specification.
- Incorporate appropriate details from the Code to ensure that the User Story are clear and unambiguous.    Refer to the core goals section to identify details necessary to understand how the core user need / goal is being facilitated and/or what behavior is occurring.    All details MUST be focused on the main goal of the User Story.    Do NOT make up information.    ALL information must be from the provided Code.
- The User Story's description uses this format as a guideline:
* As a [type of user], I want to [action or goal] so that [reason or benefit].
* For example: As a frequent traveler, I want to be able to filter hotel search results by distance from a specific landmark, so that I can find accommodations close to my desired location.
C) The observable behavior in your code is divided into steps in the order of trigger condition → system action → data flow/state change → output/feedback, making sure that at each step you can find a function, method, API call, validation logic, or data structure in your code.
- The user story's flow uses this format as a guideline:
* Step 1: [Entry: route/handler/function] receives [method/path or function call] with [inputs].\n Step 2: [Validation module] checks [rules] and returns [error] on failure.\n Step 3: [Service] performs [core logic] and accesses [DB/Repo/externals]; [key computations].\n Step 4: Writes to [storage/cache/queue], records [logs/metrics].\n Step 5: Returns [status/body]; on [error conditions] returns [mapped status/messages].
* Return flow as a single plain text string (no objects, no arrays). Use bullet lines starting with '- ' separated by newlines.
D) Extract the "how to" constraints visible in the code into verifiable non-functional requirements, including robustness, security, performance, concurrency, observability, compatibility, and compliance. Still need to be "based on code evidence only"
- The non-functional requirements uses this format as a guideline:
* -Security: [auth/role checks].\n  -Validation: [input limits/schema].\n  -Performance: [pagination/index/cache/limits/timeout].\n  -Reliability: [transactions/retries/rollback].\n  -Concurrency: [locks/unique keys/idempotency].\n  -Observability: [structured logs/metrics/tracing].\n  -Compatibility: [API version/content-type].
* Return notf as a single plain text string (no objects, no arrays). Use bullet lines starting with '- ' separated by newlines.
Please return the response in the following JSON format:
{{
"description": "User Story description",
"flow": "User Story events",
"notf":"Non-functional requirements"
}}
"""

merge_userstory_prompt = """
You are a professional software requirements analyst. Please follow these rules to analyze and merge requirements:
# feature list:
{feature_list}
# Current system macro-features list:
{module_list}
# TASK:
use the following steps to summary:
- Extract core operation objects (usually nouns) from each sub-requirement;
- Identify semantic relationships between objects (synonyms, hierarchical relations);
- For features involving different user roles (e.g., "Administrator deletes comment" vs "User adds comment"),  maintain the core module name while ensuring the implementation supports role differentiation.
- Merge features with same/core-related objects into one macro-features;
- The generated macro-features cannot already be present in the current system macro-features list;
- example: "User adds comment" and "User edits comment" can be merged into "comments management"
Please return the response in the following JSON format:
{{
"description": "macro-feature description"
}}
"""

class usecase(BaseModel):
    description: str
    flow: str
    notf: str

class module(BaseModel):
    description: str

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

def normalize_to_str(value: Any) -> str:
    """将可能为 str/dict/list/其他 的值规范化为多行字符串"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            k_str = str(k).strip()
            v_str = normalize_to_str(v)
            if k_str and v_str:
                lines.append(f"- {k_str}: {v_str}")
            elif k_str:
                lines.append(f"- {k_str}")
            elif v_str:
                lines.append(f"- {v_str}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            item_str = normalize_to_str(item)
            for line in (item_str.splitlines() or [""]):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("- "):
                    lines.append(line)
                else:
                    lines.append(f"- {line}")
        return "\n".join(lines)
    return str(value).strip()

def clean_json_text(text: str) -> str:
    txt = text.strip()
    txt = txt.replace("```json", "```")
    if txt.startswith("```") and txt.endswith("```"):
        txt = txt[3:-3].strip()

    start = txt.find("{")
    if start == -1:
        return txt
    depth = 0
    end = -1
    for i, ch in enumerate(txt[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end != -1:
        return txt[start:end+1].strip()
    return txt[start:].strip()

def parse_usecase_payload(json_str: str) -> Dict[str, Any]:
    raw = clean_json_text(json_str)
    data = json.loads(raw)

    if "description" in data and not isinstance(data["description"], str):
        data["description"] = normalize_to_str(data["description"])
    if "flow" in data and not isinstance(data["flow"], str):
        data["flow"] = normalize_to_str(data["flow"])
    if "notf" in data and not isinstance(data["notf"], str):
        data["notf"] = normalize_to_str(data["notf"])

    data.setdefault("description", "")
    data.setdefault("flow", "")
    data.setdefault("notf", "")

    return data

def generate_feature_description(feature_list: List[Feature], modelname: str):
    # 初始化特征描述
    for feature in feature_list:
        feature.feature_desc = ""

    for feature in feature_list:
        while feature.feature_desc == "":
            code = ""
            for function in feature.feature_func_list:
                code += f"function name:{function.func_fullName}\ndescription:{function.func_desc}\nflow:{function.func_flow}\nNon-functional requirements:{function.func_notf}\n"
            prompt = userstory_prompt.format(code_content=code)
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
                json_str = response.choices[0].message.content or ""
                data = parse_usecase_payload(json_str)
                result = usecase.model_validate(data)
                feature.feature_desc = result.description
                feature.feature_flow = result.flow
                feature.feature_notf = result.notf
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}\nRaw: {json_str}")
            except ValidationError as e:
                print(f"Pydantic验证失败: {e}\nRaw: {json_str}\nParsed: {data}")
            except Exception as e:
                print(f"其他错误: {e}")
            print(f"Feature ID: {feature.feature_id}, Description: {feature.feature_desc}, Flow: {feature.feature_flow}, Non-functional requirements: {feature.feature_notf}")

def merge_features_by_method_cluster(features: List[Feature], method_clusters: List[method_Cluster], modelname: str):
    merged_features_des = []
    for method_cluster in method_clusters:
        related_features = [f for f in features if f.cluster_id == method_cluster.cluster_id]
        feature_list = ""
        for i, feature in enumerate(related_features):
            feature_list += f"{i+1}. {feature.feature_desc}\n"
        module_list = ""
        for i, merged_feature_des in enumerate(merged_features_des):
            module_list += f"{i+1}. {merged_feature_des}\n"
        prompt = merge_userstory_prompt.format(
            feature_list=feature_list,
            module_list=module_list
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
            result_dict = json.loads(json_str)
            if "description" not in result_dict and isinstance(result_dict, dict):
                result_dict = {"description": next(iter(result_dict.values()))}
            result = module.model_validate(result_dict)
            merged_features_des.append(result.description)
            method_cluster.cluster_desc = result.description
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
        except ValidationError as e:
            print(f"Pydantic验证失败: {e}")
        except Exception as e:
            print(f"其他错误: {e}")
        print(f"Module ID: {method_cluster.cluster_id}, Description: {method_cluster.cluster_desc}")

def features_to_csv(features: List[Feature], method_clusters: List[method_Cluster], filename: str):
    rows = []
    for feature in features:
        method_cluster = next((mc for mc in method_clusters if mc.cluster_id == feature.cluster_id), None)
        for function in feature.feature_func_list:
            rows.append({ 
                "id": feature.feature_id,
                "cluster_id": feature.cluster_id,
                "module_desc": method_cluster.cluster_desc if method_cluster else "",
                "desc": feature.feature_desc,
                "method_name": function.func_fullName,
                "flow": feature.feature_flow,
                "notf": feature.feature_notf,
            })
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Features saved to {filename}")


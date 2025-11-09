import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from structure_analsis.java.java_import_analyzer import JavaImportAnalyzer
from structure_analsis.java.java_method_analyzer import JavaMethodAnalyzer
from structure_analsis.python.ENRE_py.enre.__main__ import main as enre_main
from model.models import Function, method_Cluster
from utils.file_operations import create_directory_summary, add_functions_to_files
from utils.file_clustering import find_best_resolution, save_to_file_cluster
from utils.function_clustering import cluster_all_functions_to_features, set_func_adj_matrix
from utils.feature_generation import generate_feature_description, merge_features_by_method_cluster, features_to_csv
from utils.method_summary import method_summary



def main(project_root: str, output_dir: str):
    # 可选择本地地址或者网页地址（github地址），这里使用本地地址
    if(project_root.startswith("http")):
        print("使用网页地址")
        # 从网页上拉取项目放在repository目录下
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repository_dir = os.path.join(script_dir, "repository")
        os.makedirs(repository_dir, exist_ok=True)
        repo_name = project_root.split("/")[-1].rstrip(".git")  # 移除 .git 后缀（如果有）
        repo_path = os.path.join(repository_dir, repo_name)
        
        # 检查目录是否已存在
        if os.path.exists(repo_path) and os.listdir(repo_path):
            print(f"项目目录已存在，使用已有目录: {repo_path}")
        else:
            # 执行 git clone 并检查返回值
            exit_code = os.system(f"git clone {project_root} {repo_path}")
            if exit_code != 0:
                print(f"错误: Git clone 失败 (退出码: {exit_code})")
                return
            # 检查克隆是否成功（目录是否存在且有内容）
            if not os.path.exists(repo_path) or not os.listdir(repo_path):
                print(f"错误: 克隆后目录为空或不存在: {repo_path}")
                return
        
        project_root = repo_path
        print(f"项目已拉取到{repo_path}")
    else:
        print("使用本地地址")
        project_root = os.path.abspath(project_root)

    # 检测项目类型
    has_java = any(f.endswith('.java') for root, _, files in os.walk(project_root) for f in files)
    has_python = any(f.endswith('.py') for root, _, files in os.walk(project_root) for f in files)
    
    # 项目结构分析
    if has_java:
        print("分析Java项目")
        file_analyzer = JavaImportAnalyzer()
        file_analyzer.analyze_project(project_root, output_dir)
        method_analyzer = JavaMethodAnalyzer()
        method_analyzer.analyze_project(project_root, output_dir)
    elif has_python:
        print("分析Python项目")
        # 使用ENRE-py分析Python项目
        enre_main([project_root, output_dir])
    else:
        print("未找到支持的代码文件（.java 或 .py）")
        return

    # 生成函数描述,可选择用函数名(function_name)/CodeT5(code_t5)/LLM(llm)生成
    language = "python" if has_python else "java"
    functions = method_summary(output_dir, strategy="code_t5", language=language)
    # 生成文件描述，在这里固定使用文件名
    files = create_directory_summary(project_root)
    add_functions_to_files(files, functions, language=language)
    # 可选择是否并行，在服务器上运行不确定是否可行，建议关闭
    is_parallel = False

    # 打印一些文件和函数信息
    for file in files[:5]:
        print(f"File ID: {file.file_id}, Name: {file.file_name}, Path: {file.file_path}, Description: {file.file_desc}")
        for function in file.func_list[:5]:
            print(f"  Function ID: {function.func_id}, Name: {function.func_name}, Description: {function.func_desc}")
        print("\n")
    
    # 生成文本向量
    model = SentenceTransformer('all-mpnet-base-v2')
    for file in files:
        file.file_txt_vector = model.encode(file.file_desc).tolist()

    # 文件聚类
    best_gamma, best_labels, results = find_best_resolution(
        files,
        a=0.5,
        n_points=25,
        gamma_min=0.01, gamma_max=0.4,
        seeds_per_gamma=8,
        use_knn=True, knn_k=20,
        use_threshold=False, threshold_tau=0.0,
        min_clusters=3, max_clusters_ratio=0.15,
        min_cluster_size=3,
        use_silhouette=False,
    )

    clusters = save_to_file_cluster(files, best_labels)

    for c in clusters:
        print(f"Cluster ID: {c.cluster_id}, {len(c.cluster_file_list)} Files: {[file.file_name for file in c.cluster_file_list]}")

    # 将clusters展开到函数层
    method_clusters = []
    for cluster in clusters:
        func_list = []
        for file in cluster.cluster_file_list:
            for function in file.func_list:
                function.func_txt_vector = model.encode(function.func_desc).tolist()
                func_list.append(function)
        method_cluster = method_Cluster(cluster.cluster_id, "", func_list)
        method_clusters.append(method_cluster)

    for method_cluster in method_clusters:
        print(f"Cluster ID: {method_cluster.cluster_id}, Functions: {[f.func_fullName for f in method_cluster.cluster_func_list]}")

    # 函数聚类
    feature_list, summary = cluster_all_functions_to_features(
        method_clusters,
        weight_parameter=0.25,
        gamma_min=0.05, gamma_max=0.5, n_points=24,
        seeds_per_gamma=8,
        use_knn=True, knn_k=20,
        use_threshold=False, threshold_tau=0.0,
        min_clusters=2, max_clusters_ratio=0.4,
        use_silhouette=False, silhouette_sample_size=None,
        objective="CPM",
        consensus_tau=0.6, consensus_gamma=0.1,
        rng_seed=2025,
        target_total_features=None,
    )
    print(f"Total Features: {len(feature_list)}")
    for f in feature_list:
        #print(f"Feature ID {f.feature_id}: {f.cluster_id} {set(x.func_file for x in f.feature_func_list)}")
        print(f"Feature ID {f.feature_id}: {[function.func_fullName for function in f.feature_func_list]}")
    
    modelname = "deepseek-v3"
    # 生成特征描述
    generate_feature_description(feature_list, modelname=modelname)

    # 合并特征
    merge_features_by_method_cluster(feature_list, method_clusters, modelname=modelname)

    # 保存到CSV
    features_to_csv(feature_list, method_clusters, os.path.join(output_dir, "features.csv"))

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = "E:/Projects/dataset/mrjob-master/mrjob"
    output_dir = os.path.join(here, "out")
    main(
        project_root=project_root,
        output_dir=output_dir
    )

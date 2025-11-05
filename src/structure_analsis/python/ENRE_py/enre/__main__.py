import argparse
import csv
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

# 将父目录添加到 sys.path，以便能够导入 enre 模块
# 获取当前文件的目录（enre 目录）
current_dir = Path(__file__).parent
# 获取父目录（ENRE-py 目录）
parent_dir = current_dir.parent
# 将父目录添加到 sys.path
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from enre.analysis.analyze_manager import AnalyzeManager
from enre.cfg.Resolver import Resolver
from enre.cfg.module_tree import Scene
from enre.passes.aggregate_control_flow_info import aggregate_cfg_info
from enre.vis.representation import DepRepr
from enre.vis.summary_repr import from_summaries, call_graph_representation
from enre.ent.EntKind import EntKind


def main(args=None) -> None:
    """
    主函数，支持从命令行调用和从 Python 代码调用
    
    Args:
        args: 可选参数列表，如果为 None 则从 sys.argv 读取
              如果提供列表，则使用该列表作为参数（例如: [project_root, output_dir]）
    """
    # 如果提供了参数列表，临时修改 sys.argv
    original_argv = sys.argv[:]
    if args is not None:
        # 将 args 列表转换为 sys.argv 格式
        sys.argv = [sys.argv[0]] + [str(arg) for arg in args]
    
    # 先提取 output_dir（第二个位置参数），避免 argparse 报错
    output_dir = None
    if args is not None and len(args) >= 2:
        output_dir = Path(args[1]) if args[1] else None
        # 从 sys.argv 中移除第二个参数，避免 argparse 报错
        if len(sys.argv) > 2:
            sys.argv = sys.argv[:2]
    elif len(sys.argv) > 2:
        # 如果第二个参数不是以 -- 开头的选项，则视为 output_dir
        if not sys.argv[2].startswith('--'):
            output_dir = Path(sys.argv[2])
            # 临时移除第二个参数，避免 argparse 报错
            sys.argv = sys.argv[:2] + sys.argv[3:]
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("root path", type=str, nargs='?',
                            help="root package path")
        parser.add_argument("--profile", action="store_true", help="output consumed time in json format")
        parser.add_argument("--cfg", action="store_true",
                            help="run control flow analysis and output module summaries")
        parser.add_argument("--compatible", action="store_true", help="output compatible format")
        parser.add_argument("--builtins", action="store", help="builtins module path")
        parser.add_argument("--cg", action="store_true", help="dump call graph in json")
        config = parser.parse_args()
        
        # 获取 root_path
        if args is not None and len(args) >= 1:
            root_path = Path(args[0]).resolve()
        elif len(sys.argv) > 1:
            root_path = Path(sys.argv[1]).resolve()
        else:
            print("错误: 需要提供项目根路径")
            return
        
        start = time.time()
        manager = enre_wrapper(root_path, config.compatible, config.cfg, config.cg, config.builtins, output_dir=output_dir)
        end = time.time()

        if config.profile:
            time_in_json = json.dumps({
                "analyzed files": len(manager.root_db.tree),
                "analysing time": end - start})
            print(time_in_json)
            # print(f"analysing time: {end - start}s")
    finally:
        # 恢复原始的 sys.argv
        sys.argv = original_argv


def dump_call_graph(project_name: str, resolver: Resolver) -> None:
    call_graph = call_graph_representation(resolver)
    out_path = f"{project_name}-call-graph-enre.json"
    with open(out_path, "w") as file:
        json.dump(call_graph, file, indent=4)


def enre_wrapper(root_path: Path, compatible_format: bool, need_cfg: bool, need_call_graph: bool,
                 builtin_module: str, output_dir: Path = None) -> AnalyzeManager:
    project_name = root_path.name
    builtins_path = Path(builtin_module) if builtin_module else None
    manager = AnalyzeManager(root_path, builtins_path)
    manager.work_flow()
    out_path = Path(f"{project_name}-report-enre.json")
    if need_cfg:
        print("dependency analysis finished, now running control flow analysis")
        resolver = cfg_wrapper(root_path, manager.scene)
        print("control flow analysis finished")
        aggregate_cfg_info(manager.root_db, resolver)
        if need_call_graph:
            dump_call_graph(project_name, resolver)

    with open(out_path, "w") as file:
        if not compatible_format:
            json.dump(DepRepr.from_package_db(manager.root_db).to_json_1(), file, indent=4)
        else:
            repr = DepRepr.from_package_db(manager.root_db).to_json()
            json.dump(repr, file, indent=4)

    # 生成 CSV 文件
    dep_repr = DepRepr.from_package_db(manager.root_db)
    # root_path 是分析时的项目根目录，应该包含源文件
    generate_csv_files(dep_repr, manager.root_db, root_path, output_dir)

    return manager


def cfg_wrapper(root_path: Path, scene: Scene) -> Resolver:
    resolver = Resolver(scene)
    resolver.resolve_all()
    out_path = Path(f"{root_path.name}-report-cfg.txt")
    with open(out_path, "w") as file:
        summary_repr = from_summaries(scene.summaries)
        file.write(summary_repr)
    return resolver


def extract_method_code(file_path: str, start_line: int, end_line: int, project_root: Path = None) -> str:
    """从文件中提取指定行号的代码"""
    if start_line <= 0:
        return ""
    
    try:
        # 尝试多种路径解析方式
        path = None
        tried_paths = []
        
        # 1. 尝试直接路径（绝对路径或相对当前目录）
        test_path = Path(file_path)
        tried_paths.append(str(test_path.absolute()))
        if test_path.exists():
            path = test_path
        
        # 2. 尝试相对于项目根目录的路径
        if not path and project_root:
            potential_path = project_root / file_path
            tried_paths.append(str(potential_path.absolute()))
            if potential_path.exists():
                path = potential_path
        
        # 3. 尝试标准化路径分隔符
        if not path and project_root:
            # 根据项目根目录的路径分隔符来标准化
            if "\\" in str(project_root):
                normalized_path = file_path.replace("/", "\\")
            else:
                normalized_path = file_path.replace("\\", "/")
            potential_path = project_root / normalized_path
            tried_paths.append(str(potential_path.absolute()))
            if potential_path.exists():
                path = potential_path
        
        if not path or not path.exists():
            # 如果所有路径都失败，尝试查找文件（不区分大小写）
            if project_root:
                file_name = Path(file_path).name
                for py_file in project_root.rglob("*.py"):
                    if py_file.name.lower() == file_name.lower():
                        # 检查路径是否匹配（忽略大小写和路径分隔符）
                        rel_path = str(py_file.relative_to(project_root)).replace("\\", "/")
                        if rel_path.lower() == file_path.lower().replace("\\", "/"):
                            path = py_file
                            break
        
        if not path or not path.exists():
            # 如果文件不存在，可能是因为文件路径是相对于分析时的项目根目录的
            # 尝试在项目根目录下递归查找文件名匹配的文件
            if project_root and project_root.exists():
                file_name = Path(file_path).name
                # 只在项目根目录下查找，避免搜索整个系统
                for py_file in project_root.rglob(file_name):
                    # 检查相对路径是否匹配（忽略大小写和路径分隔符）
                    rel_path = str(py_file.relative_to(project_root)).replace("\\", "/")
                    expected_rel_path = file_path.replace("\\", "/")
                    if rel_path.lower() == expected_rel_path.lower():
                        path = py_file
                        break
                
                # 如果还是找不到，尝试只匹配文件名
                if (not path or not path.exists()) and file_name:
                    for py_file in project_root.rglob(file_name):
                        path = py_file
                        break
            
            if not path or not path.exists():
                return ""
        
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            if start_line > len(lines):
                return ""
            
            # 如果 end_line 无效（<=0 或 -1），尝试提取到函数结束
            # 首先尝试找到函数的结束位置（通过查找下一个相同或更小缩进的非空行）
            if end_line <= 0 or end_line > len(lines):
                # 如果没有有效的 end_line，尝试从 start_line 开始找到函数的结束
                # 获取函数定义行的缩进
                if start_line <= len(lines):
                    def_line = lines[start_line - 1]
                    # 计算缩进（假设使用空格或制表符）
                    indent_level = len(def_line) - len(def_line.lstrip())
                    
                    # 从下一行开始查找，直到找到相同或更小缩进的非空行
                    end_line = start_line
                    for i in range(start_line, len(lines)):
                        line = lines[i]
                        if line.strip():  # 非空行
                            line_indent = len(line) - len(line.lstrip())
                            if line_indent <= indent_level:
                                end_line = i  # 找到函数结束，但不包括这一行
                                break
                    else:
                        # 如果没找到，就提取到文件末尾
                        end_line = len(lines)
            
            # 确保 end_line 在有效范围内
            if end_line > len(lines):
                end_line = len(lines)
            if end_line < start_line:
                end_line = start_line
            
            # Python 行号从1开始，数组索引从0开始
            code_lines = lines[start_line - 1:end_line]
            return "".join(code_lines).rstrip()
    except Exception as e:
        # 调试：可以取消注释来查看错误
        # print(f"Error extracting code from {file_path}: {e}")
        return ""


def extract_method_code_from_entity(node, package_db, project_root: Path = None) -> str:
    """从原始 Entity 中尝试提取方法代码（优先方法）"""
    from enre.ent.EntKind import EntKind
    
    # 尝试从 package_db 中找到对应的 Entity
    func_entity = None
    for rel_path, module_db in package_db.tree.items():
        for ent in module_db.dep_db.ents:
            if ent.id == node.id and ent.kind() == EntKind.Function:
                func_entity = ent
                break
        if func_entity:
            break
    
    if not func_entity:
        # 如果找不到，也尝试全局数据库
        for ent in package_db.global_db.ents:
            if ent.id == node.id and ent.kind() == EntKind.Function:
                func_entity = ent
                break
    
    if not func_entity:
        return ""
    
    # 使用 Entity 的位置信息（更准确）
    try:
        file_path = str(func_entity.location.file_path)
        # 确保路径使用正斜杠（与 Node 中的格式一致）
        file_path = file_path.replace("\\", "/")
        start_line = func_entity.location.code_span.start_line
        end_line = func_entity.location.code_span.end_line
        
        # 如果 Entity 的路径是绝对路径，尝试使用它
        code = extract_method_code(file_path, start_line, end_line, project_root)
        
        # 如果提取失败，尝试使用相对路径（从 Node 中获取）
        if not code and node.file_path != file_path:
            code = extract_method_code(node.file_path, start_line, end_line, project_root)
        
        return code
    except Exception:
        # 如果访问 Entity 位置信息失败，回退到使用 Node 的信息
        return extract_method_code(node.file_path, node.start_line, node.end_line, project_root)


def extract_method_code_direct(node, package_db, project_root: Path = None) -> str:
    """直接从 package_db 中查找文件并提取代码"""
    from enre.ent.EntKind import EntKind
    
    # 尝试从 package_db 中找到对应的 Entity
    func_entity = None
    for rel_path, module_db in package_db.tree.items():
        for ent in module_db.dep_db.ents:
            if ent.id == node.id and ent.kind() == EntKind.Function:
                func_entity = ent
                # 同时获取模块的路径信息
                module_path = rel_path
                break
        if func_entity:
            break
    
    if not func_entity:
        return ""
    
    # 尝试多种文件路径
    file_paths_to_try = []
    
    # 1. 使用 Entity 的 file_path
    try:
        file_paths_to_try.append(str(func_entity.location.file_path))
    except:
        pass
    
    # 2. 使用 Node 的 file_path
    file_paths_to_try.append(node.file_path)
    
    # 3. 尝试使用模块路径 + 文件名
    try:
        if module_path:
            file_name = Path(node.file_path).name
            file_paths_to_try.append(str(module_path / file_name))
    except:
        pass
    
    # 4. 尝试使用项目根目录 + 相对路径
    if project_root:
        file_paths_to_try.append(str(project_root / node.file_path))
    
    start_line = func_entity.location.code_span.start_line
    end_line = func_entity.location.code_span.end_line
    
    # 如果 Entity 的行号无效，使用 Node 的行号
    if start_line <= 0:
        start_line = node.start_line
    if end_line <= 0:
        end_line = node.end_line
    
    # 尝试每个可能的路径
    for file_path in file_paths_to_try:
        if file_path:
            code = extract_method_code(file_path, start_line, end_line, project_root)
            if code:
                return code
    
    return ""


def get_function_signature_with_params(node, package_db) -> str:
    """获取包含参数信息的函数签名"""
    from enre.ent.EntKind import EntKind, RefKind
    
    # 首先尝试从 package_db 中找到对应的 Entity
    func_entity = None
    for rel_path, module_db in package_db.tree.items():
        for ent in module_db.dep_db.ents:
            if ent.id == node.id and ent.kind() == EntKind.Function:
                func_entity = ent
                break
        if func_entity:
            break
    
    if not func_entity:
        # 如果找不到，也尝试全局数据库
        for ent in package_db.global_db.ents:
            if ent.id == node.id and ent.kind() == EntKind.Function:
                func_entity = ent
                break
    
    if not func_entity:
        # 如果找不到 Entity，返回原始 longname
        return node.longname
    
    # 从 Entity 的 refs 中查找参数（DefineKind 的 Parameter 类型）
    params = []
    for ref in func_entity.refs():
        if (ref.ref_kind == RefKind.DefineKind and 
            ref.target_ent.kind() == EntKind.Parameter):
            # 参数名称从 longname 中提取（通常是 function_name.param_name）
            param_name = ref.target_ent.longname.name
            params.append((ref.lineno, ref.col_offset, param_name))
    
    # 按行号和列号排序参数（保持它们在源代码中的顺序）
    params.sort(key=lambda x: (x[0], x[1]))
    param_names = [p[2] for p in params]
    
    # 构建带参数的签名
    if param_names:
        return f"{node.longname}({', '.join(param_names)})"
    else:
        return f"{node.longname}()"


def generate_csv_files(dep_repr: DepRepr, package_db = None, project_root: Path = None, output_dir: Path = None) -> None:
    """生成 CSV 文件: files.csv, methods.csv, file_adj_matrix.csv, func_adj_matrix.csv"""
    
    # 处理 output_dir：如果为 None，使用当前目录
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集文件和方法的节点
    file_nodes = []  # Module 类型的节点
    func_nodes = []  # Function 类型的节点
    node_id_to_index = {}  # 节点ID到索引的映射
    
    # 分离文件和函数节点
    for node in dep_repr._node_list:
        if node.ent_type == EntKind.Module.value:
            file_nodes.append(node)
            node_id_to_index[node.id] = len(file_nodes) - 1
        elif node.ent_type == EntKind.Function.value:
            func_nodes.append(node)
            node_id_to_index[node.id] = len(func_nodes) - 1
    
    # 生成 files.csv到output_dir目录下
    # with open(output_dir / "files.csv", "w", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["id", "file_path", "longname", "start_line", "end_line", "start_col", "end_col"])
    #     for idx, node in enumerate(file_nodes):
    #         writer.writerow([
    #             idx,  # 使用索引从0开始
    #             node.file_path,
    #             node.longname,
    #             node.start_line,
    #             node.end_line,
    #             node.start_col,
    #             node.end_col
    #         ])
    
    # 生成 methods.csv
    with open(output_dir / "methods.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "method_signature", "func_file", "method_code"])
        for idx, node in enumerate(func_nodes):
            # 获取带参数的方法签名
            method_signature = get_function_signature_with_params(node, package_db) if package_db else node.longname
            
            # 优先从原始 Entity 获取代码（因为它有更准确的位置信息）
            method_code = ""
            if package_db:
                method_code = extract_method_code_from_entity(node, package_db, project_root)
            
            # 如果从 Entity 提取失败，尝试使用 Node 的位置信息
            if not method_code:
                method_code = extract_method_code(node.file_path, node.start_line, node.end_line, project_root)
            
            # 调试：如果还是为空，尝试其他方法
            if not method_code and package_db:
                # 尝试直接从 package_db 中查找文件
                method_code = extract_method_code_direct(node, package_db, project_root)
            
            writer.writerow([
                idx,  # 使用索引从0开始
                method_signature,  # method_signature with parameters
                node.file_path,  # func_file
                method_code  # method_code
            ])
    
    # 构建节点ID到节点的映射
    node_id_to_node = {node.id: node for node in dep_repr._node_list}
    
    # 构建文件ID到文件路径的映射（用于文件邻接矩阵）
    file_id_to_path = {node.id: node.file_path for node in file_nodes}
    
    # 构建函数ID到文件路径的映射（用于根据函数依赖推导文件依赖）
    func_id_to_file_path = {}
    for node in func_nodes:
        func_id_to_file_path[node.id] = node.file_path
    
    # 构建文件邻接矩阵
    file_id_to_index = {node.id: idx for idx, node in enumerate(file_nodes)}
    file_adj_matrix = defaultdict(lambda: defaultdict(int))
    
    # 构建函数邻接矩阵
    func_id_to_index = {node.id: idx for idx, node in enumerate(func_nodes)}
    func_adj_matrix = defaultdict(lambda: defaultdict(int))
    
    # 处理所有边
    for edge in dep_repr._edge_list:
        src_id = edge.src
        dest_id = edge.dest
        
        src_node = node_id_to_node.get(src_id)
        dest_node = node_id_to_node.get(dest_id)
        
        if src_node and dest_node:
            # 文件邻接矩阵：如果两个文件之间有依赖关系，或者文件A中的函数依赖文件B中的函数
            if src_node.ent_type == EntKind.Module.value and dest_node.ent_type == EntKind.Module.value:
                # 直接的模块到模块依赖
                if src_id in file_id_to_index and dest_id in file_id_to_index:
                    src_idx = file_id_to_index[src_id]
                    dest_idx = file_id_to_index[dest_id]
                    file_adj_matrix[src_idx][dest_idx] = 1
            elif src_node.ent_type == EntKind.Function.value and dest_node.ent_type == EntKind.Function.value:
                # 函数之间的依赖，推导文件之间的依赖
                src_file_path = func_id_to_file_path.get(src_id)
                dest_file_path = func_id_to_file_path.get(dest_id)
                if src_file_path and dest_file_path and src_file_path != dest_file_path:
                    # 找到对应的文件节点ID
                    src_file_node = next((n for n in file_nodes if n.file_path == src_file_path), None)
                    dest_file_node = next((n for n in file_nodes if n.file_path == dest_file_path), None)
                    if src_file_node and dest_file_node:
                        src_file_idx = file_id_to_index[src_file_node.id]
                        dest_file_idx = file_id_to_index[dest_file_node.id]
                        file_adj_matrix[src_file_idx][dest_file_idx] = 1
            
            # 函数邻接矩阵
            if src_node.ent_type == EntKind.Function.value and dest_node.ent_type == EntKind.Function.value:
                if src_id in func_id_to_index and dest_id in func_id_to_index:
                    src_idx = func_id_to_index[src_id]
                    dest_idx = func_id_to_index[dest_id]
                    func_adj_matrix[src_idx][dest_idx] = 1
    
    # 生成 file_adj_matrix.csv
    with open(output_dir / "file_adj_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 写入矩阵
        for i, node in enumerate(file_nodes):
            row = [file_adj_matrix[i].get(j, 0) for j in range(len(file_nodes))]
            writer.writerow(row)
    
    # 生成 method_adj_matrix.csv
    with open(output_dir / "method_adj_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 写入矩阵
        for i, node in enumerate(func_nodes):
            row = [func_adj_matrix[i].get(j, 0) for j in range(len(func_nodes))]
            writer.writerow(row)

if __name__ == '__main__':
    main()

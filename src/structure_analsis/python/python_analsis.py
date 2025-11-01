import os
import csv
import ast
import re
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

class PythonMethodAnalyzer:
    def __init__(self):
        # 存储函数信息：ID -> (函数签名, 函数代码, 类/模块名, 文件名, 起始行, 结束行)
        self.methods: Dict[int, Tuple[str, str, str, str, int, int]] = {}
        # 类/模块名到方法ID的映射
        self.class_methods: Dict[str, Set[int]] = defaultdict(set)
        # 调用关系：调用者ID -> 被调用者ID集合
        self.call_relations: Dict[int, Set[int]] = defaultdict(set)
        # 存储文件AST树
        self.file_trees: Dict[str, ast.AST] = {}
        # 当前方法ID计数器
        self.current_id = 1
        # 存储导入映射（每个文件一个）
        self.import_mapping: Dict[str, Dict[str, str]] = {}
        # 存储模块名（按文件）
        self.module_names: Dict[str, str] = {}
        # 当前解析上下文（当前所在的函数ID）
        self.current_context: List[int] = []
        # 文件依赖关系：文件 -> 依赖的文件集合
        self.file_dependencies: Dict[str, Set[str]] = defaultdict(set)
        # 文件名到ID的映射
        self.file_to_id: Dict[str, int] = {}
        # 所有文件列表
        self.all_files: List[str] = []
        # 项目根目录
        self.project_root: Optional[str] = None
        # 文件到方法ID的映射（用于快速查找）
        self.file_to_method_ids: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)  # file_path -> [(method_id, start, end), ...]
    
    def analyze_project(self, project_root: str, output_dir: str = "."):
        if not os.path.isdir(project_root):
            print(f"无效的项目目录: {project_root}")
            return

        self.project_root = os.path.abspath(project_root)
        python_files = self._collect_python_files(project_root)
        print(f"找到 {len(python_files)} 个Python文件")

        # 第一遍：收集函数信息和导入关系
        print("开始解析文件信息...")
        for i, file_path in enumerate(python_files, 1):
            if i % 10 == 0 or i == len(python_files):
                print(f"  解析进度: {i}/{len(python_files)} ({i*100//len(python_files)}%)")
            self._parse_file_info(file_path)
        print(f"解析完成，共找到 {len(self.methods)} 个函数/方法")

        # 第二遍：分析调用关系
        print("开始分析调用关系...")
        for i, file_path in enumerate(python_files, 1):
            if i % 10 == 0 or i == len(python_files):
                print(f"  分析进度: {i}/{len(python_files)} ({i*100//len(python_files)}%)")
            self._analyze_call_relations(file_path)
        print(f"调用关系分析完成，共找到 {sum(len(v) for v in self.call_relations.values())} 条调用关系")

        # 输出
        print("正在生成输出文件...")
        os.makedirs(output_dir, exist_ok=True)
        self._write_method_csv(os.path.join(output_dir, "method.csv"))
        self._generate_call_matrix(os.path.join(output_dir, "method_adj_matrix.csv"))
        self._generate_file_adj_matrix(os.path.join(output_dir, "file_adj_matrix.csv"))
        print("分析完成！")
    
    def _collect_python_files(self, root_dir: str) -> List[str]:
        python_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files
    
    def _parse_file_info(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                source_code = file.read()
            
            tree = ast.parse(source_code)
            self.file_trees[file_path] = tree
            
            # 计算模块名（相对于项目根目录）
            if self.project_root:
                try:
                    rel_path = os.path.relpath(file_path, self.project_root)
                except ValueError:
                    # 如果无法计算相对路径，使用绝对路径的basename
                    rel_path = os.path.basename(file_path)
            else:
                rel_path = os.path.basename(file_path)
            
            module_name = os.path.splitext(rel_path)[0].replace(os.sep, '.')
            if module_name.endswith('.__init__'):
                module_name = module_name.replace('.__init__', '')
            if module_name.startswith('.'):
                module_name = module_name[1:]
            
            self.module_names[file_path] = module_name
            
            # 创建导入映射
            import_map: Dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_name = alias.asname if alias.asname else alias.name.split('.')[0]
                        import_map[import_name] = alias.name
                elif isinstance(node, ast.ImportFrom):
                    module_name_import = node.module or ''
                    if node.level > 0:  # 相对导入
                        base_module = module_name
                        for _ in range(node.level - 1):
                            if '.' in base_module:
                                base_module = '.'.join(base_module.split('.')[:-1])
                        if module_name_import:
                            module_name_import = f"{base_module}.{module_name_import}" if base_module else module_name_import
                    for alias in node.names:
                        import_name = alias.asname if alias.asname else alias.name
                        if module_name_import:
                            import_map[import_name] = f"{module_name_import}.{alias.name}"
                        else:
                            import_map[import_name] = alias.name
            self.import_mapping[file_path] = import_map
            
            # 收集文件列表
            if file_path not in self.file_to_id:
                file_id = len(self.all_files)
                self.file_to_id[file_path] = file_id
                self.all_files.append(file_path)
            
            # 处理类和函数
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._parse_class_methods(file_path, module_name, node, source_code)
                elif isinstance(node, ast.FunctionDef):
                    # 模块级函数
                    self._record_method(file_path, module_name, node, source_code)
                    
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"解析文件 {file_path} 时出错: {str(e)}")
    
    def _parse_class_methods(self, file_path: str, module_name: str, class_node: ast.ClassDef, source_code: str):
        class_name = f"{module_name}.{class_node.name}" if module_name else class_node.name
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                self._record_method(file_path, class_name, node, source_code, class_node)
    
    def _record_method(self, file_path: str, class_name: str, func_node: ast.FunctionDef, source_code: str, class_node: Optional[ast.ClassDef] = None):
        # 函数名
        method_name = func_node.name
        
        # 参数列表
        params = []
        for arg in func_node.args.args:
            param_name = arg.arg
            # 尝试获取参数类型注解
            if arg.annotation:
                # Python 3.9+ 支持 ast.unparse
                try:
                    if hasattr(ast, 'unparse'):
                        param_type = ast.unparse(arg.annotation)
                    else:
                        param_type = self._get_annotation_str(arg.annotation)
                    params.append(f"{param_type} {param_name}")
                except:
                    params.append(param_name)
            else:
                params.append(param_name)
        
        # 处理*args和**kwargs
        if func_node.args.vararg:
            params.append(f"*{func_node.args.vararg.arg}")
        if func_node.args.kwarg:
            params.append(f"**{func_node.args.kwarg.arg}")
        
        param_str = ", ".join(params)
        
        # 签名（对于类方法，包含self/cls参数）
        method_signature = f"{class_name}.{method_name}( {param_str} )"
        
        # 位置信息
        start_line = func_node.lineno
        end_line = func_node.end_lineno if hasattr(func_node, 'end_lineno') else self._get_method_end_line(func_node, source_code)
        
        # 源码
        method_code = self._extract_method_code(source_code, start_line, end_line)
        
        # ID 分配与存储
        method_id = self.current_id
        self.current_id += 1
        
        self.methods[method_id] = (method_signature, method_code, class_name, file_path, start_line, end_line)
        self.class_methods[class_name].add(method_id)
        # 添加文件到方法ID的映射（用于快速查找）
        self.file_to_method_ids[file_path].append((method_id, start_line, end_line))
    
    def _get_annotation_str(self, annotation) -> str:
        """将类型注解转换为字符串"""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Attribute):
            return f"{self._get_annotation_str(annotation.value)}.{annotation.attr}"
        elif isinstance(annotation, ast.Subscript):
            base = self._get_annotation_str(annotation.value)
            if isinstance(annotation.slice, ast.Index) and hasattr(annotation.slice, 'value'):
                args = self._get_annotation_str(annotation.slice.value)
            elif hasattr(annotation, 'slice'):
                args = self._get_annotation_str(annotation.slice)
            else:
                args = "?"
            return f"{base}[{args}]"
        return "?"
    
    def _get_method_end_line(self, func_node: ast.FunctionDef, source_code: str) -> int:
        """获取方法结束行号"""
        # Python 3.8+ 支持 end_lineno
        if hasattr(func_node, 'end_lineno') and func_node.end_lineno:
            return func_node.end_lineno
        
        start_line = func_node.lineno
        lines = source_code.split('\n')
        
        # 查找函数体的结束位置
        if func_node.body:
            last_stmt = func_node.body[-1]
            if hasattr(last_stmt, 'end_lineno') and last_stmt.end_lineno:
                return last_stmt.end_lineno
        
        # 如果没有end_lineno，尝试通过缩进推断
        if len(lines) >= start_line:
            if start_line > 0:
                start_indent = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
                for i in range(start_line, len(lines)):
                    line = lines[i]
                    if line.strip() and len(line) - len(line.lstrip()) <= start_indent:
                        return i + 1
            return len(lines)
        
        return start_line
    
    def _extract_method_code(self, source_code: str, start_line: int, end_line: int) -> str:
        """按行号提取方法代码"""
        lines = source_code.split('\n')
        if start_line <= 0:
            return ""
        if end_line <= 0 or end_line < start_line:
            end_line = start_line
        start_idx = max(0, min(start_line - 1, len(lines) - 1))
        end_idx = max(start_idx, min(end_line - 1, len(lines) - 1))
        return '\n'.join(lines[start_idx:end_idx + 1])
    
    def _analyze_call_relations(self, file_path: str):
        """分析调用关系"""
        try:
            tree = self.file_trees[file_path]
            module_name = self.module_names[file_path]
            import_map = self.import_mapping[file_path]
            
            # 递归遍历所有节点
            self._traverse_and_analyze(tree, file_path, module_name, import_map)
            
        except Exception as e:
            print(f"分析调用关系时出错: {file_path} - {str(e)}")
    
    def _traverse_and_analyze(self, node, file_path: str, module_name: str, import_map: Dict[str, str]):
        """递归遍历AST节点并分析调用关系"""
        # 更新上下文
        if hasattr(node, 'lineno'):
            self._update_context(file_path, node.lineno)
        
        # 处理函数调用
        if isinstance(node, ast.Call):
            self._process_function_call(file_path, node, module_name, import_map)
        
        # 处理导入（用于文件依赖关系）
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self._process_import_for_dependencies(file_path, node, module_name)
        
        # 递归遍历子节点
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self._traverse_and_analyze(item, file_path, module_name, import_map)
            elif isinstance(value, ast.AST):
                self._traverse_and_analyze(value, file_path, module_name, import_map)
    
    def _update_context(self, file_path: str, line: int):
        """根据位置更新当前解析上下文（优化版：使用文件索引）"""
        if not line:
            self.current_context = []
            return
        
        # 优化：只检查当前文件的方法，使用文件索引而不是遍历所有方法
        matching_methods = []
        
        # 使用文件索引快速查找当前文件的方法
        if file_path in self.file_to_method_ids:
            for method_id, start, end in self.file_to_method_ids[file_path]:
                if start <= line <= end:
                    matching_methods.append((method_id, start, end))
        else:
            # 回退：如果索引不存在，遍历所有方法（应该不会发生，但为了安全）
            for method_id, (_, _, _, m_file_path, start, end) in self.methods.items():
                if m_file_path == file_path and start <= line <= end:
                    matching_methods.append((method_id, start, end))
        
        if not matching_methods:
            self.current_context = []
            return
        
        # 按嵌套深度排序（最内层的方法在最前面）
        matching_methods.sort(key=lambda x: (x[2] - x[1], -x[1]), reverse=True)
        self.current_context = [m[0] for m in matching_methods]
    
    def _process_function_call(self, file_path: str, node: ast.Call, module_name: str, import_map: Dict[str, str]):
        if not self.current_context:
            return
        
        caller_id = self.current_context[0]
        
        # 解析被调用的函数
        func_name = None
        target_module = None
        
        if isinstance(node.func, ast.Name):
            # 直接函数调用 func()
            func_name = node.func.id
            # 检查是否是导入的函数
            if func_name in import_map:
                full_name = import_map[func_name]
                if '.' in full_name:
                    parts = full_name.rsplit('.', 1)
                    target_module = parts[0]
                    func_name = parts[1]
                else:
                    target_module = None
        elif isinstance(node.func, ast.Attribute):
            # 属性调用 obj.method() 或 module.func()
            func_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                # module.func() 或 obj.method()
                value_name = node.func.value.id
                if value_name in import_map:
                    # 是导入的模块
                    target_module = import_map[value_name]
                elif value_name == "self":
                    # 实例方法调用，使用调用者的类
                    caller_class = self.methods[caller_id][2]
                    target_module = caller_class
                else:
                    # 可能是局部变量或字段，尝试推断类型
                    target_module = self._infer_expression_type(caller_id, value_name, file_path, module_name, import_map)
        
        # 获取参数数量
        arg_count = len(node.args) + (1 if node.keywords else 0)
        
        # 查找匹配的函数
        callee_ids = []
        
        if target_module:
            # 在目标模块/类中查找
            callee_ids = self._find_matching_methods(target_module, func_name, arg_count)
        
        # 如果找不到，尝试模糊匹配（限制搜索范围以提高性能）
        if not callee_ids:
            # 在当前类/模块中查找
            caller_class = self.methods[caller_id][2]
            for method_id in self.class_methods.get(caller_class, []):
                signature = self.methods[method_id][0]
                if self._simple_method_match(signature, func_name, arg_count):
                    if method_id not in callee_ids:
                        callee_ids.append(method_id)
            
            # 只在当前文件的模块中查找，而不是所有类/模块（性能优化）
            # 这样避免在大项目中遍历所有函数
            if not callee_ids and module_name:
                # 尝试从当前模块名查找
                if module_name in self.class_methods:
                    for method_id in self.class_methods[module_name]:
                        signature = self.methods[method_id][0]
                        if self._simple_method_match(signature, func_name, arg_count):
                            if method_id not in callee_ids:
                                callee_ids.append(method_id)
            
            # 最后才在所有已知类/模块中查找（性能最差，仅作为备选）
            # 如果项目很大，可以注释掉这部分以减少搜索时间
            if not callee_ids and len(self.methods) < 10000:  # 只在方法数量不太大时搜索
                for class_name in self.class_methods:
                    if len(callee_ids) >= 10:  # 限制匹配数量
                        break
                    for method_id in self.class_methods[class_name]:
                        signature = self.methods[method_id][0]
                        if self._simple_method_match(signature, func_name, arg_count):
                            if method_id not in callee_ids:
                                callee_ids.append(method_id)
                                if len(callee_ids) >= 10:
                                    break
        
        # 记录调用关系
        for callee_id in callee_ids:
            if callee_id != caller_id:
                self.call_relations[caller_id].add(callee_id)
    
    def _infer_expression_type(self, caller_id: int, var_name: str, file_path: str, module_name: str, import_map: Dict[str, str]) -> Optional[str]:
        """推断变量类型（简化）"""
        # 检查是否是导入的模块
        if var_name in import_map:
            return import_map[var_name]
        
        # 检查是否是当前类的实例
        caller_class = self.methods[caller_id][2]
        if var_name == "self":
            return caller_class
        
        return None
    
    def _find_matching_methods(self, class_name: str, method_name: str, arg_count: int) -> List[int]:
        """查找匹配的方法"""
        method_ids = []
        
        if class_name in self.class_methods:
            for method_id in self.class_methods[class_name]:
                signature = self.methods[method_id][0]
                if self._simple_method_match(signature, method_name, arg_count):
                    method_ids.append(method_id)
        
        return method_ids
    
    def _simple_method_match(self, signature: str, method_name: str, arg_count: int) -> bool:
        """简单的方法名匹配（不考虑类型）"""
        # 检查方法名是否在签名中
        pattern = f".{method_name}("
        if pattern not in signature:
            return False
        
        # 提取参数部分
        match = re.search(r'\((.*?)\)', signature)
        if not match:
            return arg_count == 0
        
        params_str = match.group(1).strip()
        if not params_str:
            return arg_count == 0
        
        # 计算参数数量（过滤掉*args和**kwargs）
        param_list = [p.strip() for p in params_str.split(',') if p.strip() and not p.strip().startswith('*')]
        # 简化：Python可能有默认参数，所以只检查最小参数数量
        return len(param_list) <= arg_count or arg_count == 0
    
    def _process_import_for_dependencies(self, file_path: str, node, module_name: str):
        """处理导入以构建文件依赖关系"""
        try:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_module = alias.name
                    target_file = self._resolve_module_to_file(imported_module)
                    if target_file and target_file != file_path:
                        self.file_dependencies[file_path].add(target_file)
            elif isinstance(node, ast.ImportFrom):
                imported_module = node.module or ''
                if node.level > 0:  # 相对导入
                    base_module = module_name
                    for _ in range(node.level - 1):
                        if '.' in base_module:
                            base_module = '.'.join(base_module.split('.')[:-1])
                    if imported_module:
                        imported_module = f"{base_module}.{imported_module}" if base_module else imported_module
                
                if imported_module:
                    target_file = self._resolve_module_to_file(imported_module)
                    if target_file and target_file != file_path:
                        self.file_dependencies[file_path].add(target_file)
        except Exception as e:
            pass
    
    def _resolve_module_to_file(self, module_name: str) -> Optional[str]:
        """将模块名解析为文件路径"""
        # 在已知的模块名中查找
        for file_path, mod_name in self.module_names.items():
            if mod_name == module_name or mod_name.endswith('.' + module_name):
                return file_path
            # 检查是否是模块的一部分
            if module_name.startswith(mod_name + '.'):
                return file_path
        return None
    
    def _write_method_csv(self, output_file: str):
        """写入方法信息到 CSV"""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ID', 'method_signature', 'method_code', 'class_name', 'file_path', 'start_line', 'end_line'])
            for method_id, (signature, code, class_name, file_path, start_line, end_line) in self.methods.items():
                clean_code = code.replace('\n', '\\n').replace('\r', '\\r')
                writer.writerow([method_id, signature, clean_code, class_name, file_path, start_line, end_line])
        print(f"方法信息已写入: {output_file}")
    
    def _generate_call_matrix(self, output_file: str):
        """生成调用关系邻接矩阵"""
        if not self.methods:
            print("未找到可分析的方法")
            return
        
        method_ids = sorted(self.methods.keys())
        id_index = {method_id: idx for idx, method_id in enumerate(method_ids)}
        size = len(method_ids)
        
        adj_matrix = [[0] * size for _ in range(size)]
        
        for caller_id, callee_ids in self.call_relations.items():
            if caller_id in id_index:
                caller_idx = id_index[caller_id]
                for callee_id in callee_ids:
                    if callee_id in id_index:
                        callee_idx = id_index[callee_id]
                        adj_matrix[caller_idx][callee_idx] = 1
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Caller/Callee'] + [str(mid) for mid in method_ids])
            for i, method_id in enumerate(method_ids):
                row = [str(method_id)] + adj_matrix[i]
                writer.writerow(row)
        print(f"方法调用邻接矩阵已写入: {output_file}")
    
    def _generate_file_adj_matrix(self, output_file: str):
        """生成文件依赖关系邻接矩阵"""
        if not self.all_files:
            print("未找到可分析的文件")
            return
        
        # 创建文件路径到索引的映射
        file_index = {file_path: idx for idx, file_path in enumerate(self.all_files)}
        size = len(self.all_files)
        
        adj_matrix = [[0] * size for _ in range(size)]
        
        for src_file, deps in self.file_dependencies.items():
            if src_file in file_index:
                src_idx = file_index[src_file]
                for dep_file in deps:
                    if dep_file in file_index:
                        dep_idx = file_index[dep_file]
                        adj_matrix[src_idx][dep_idx] = 1
        
        # 使用文件名作为标识
        file_names = [os.path.basename(f) for f in self.all_files]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([''] + file_names)
            for i, file_name in enumerate(file_names):
                row = [file_name] + adj_matrix[i]
                writer.writerow(row)
        print(f"文件依赖邻接矩阵已写入: {output_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python python_analsis.py <project_root> [output_dir]")
        sys.exit(1)
    
    project_root = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    analyzer = PythonMethodAnalyzer()
    analyzer.analyze_project(project_root, output_dir)

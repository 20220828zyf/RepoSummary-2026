import os
import csv
import re
import javalang
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

class JavaMethodAnalyzer:
    def __init__(self):
        # 存储方法信息：ID -> (方法签名, 方法代码, 类名, 文件名, 起始行, 结束行)
        self.methods: Dict[int, Tuple[str, str, str, str, int, int]] = {}
        # 类名到方法ID的映射
        self.class_methods: Dict[str, Set[int]] = defaultdict(set)
        # 调用关系：调用者ID -> 被调用者ID集合
        self.call_relations: Dict[int, Set[int]] = defaultdict(set)
        # 存储文件AST树
        self.file_trees: Dict[str, Any] = {}
        # 当前方法ID计数器
        self.current_id = 1
        # 存储导入映射（每个文件一个）
        self.import_mapping: Dict[str, Dict[str, str]] = {}
        # 存储包名（按文件）
        self.package_names: Dict[str, str] = {}
        # 类继承/实现关系：类或接口 -> [父类/父接口...]
        self.class_hierarchy: Dict[str, List[str]] = defaultdict(list)
        # 方法签名到ID的映射
        self.signature_to_id: Dict[str, List[int]] = defaultdict(list)
        # 当前解析上下文（当前所在的方法ID栈，支持嵌套）
        self.current_context: List[int] = []
        # 存储字段类型信息（类名 -> 字段名 -> 类型）
        self.field_types: Dict[str, Dict[str, str]] = defaultdict(dict)
        # 存储局部变量类型信息（方法ID -> 变量名 -> 类型）
        self.local_vars: Dict[int, Dict[str, str]] = defaultdict(dict)
        # 存储方法的返回类型
        self.method_return_types: Dict[int, Optional[str]] = {}
    
    def analyze_project(self, project_root: str, output_dir: str = "."):
        if not os.path.isdir(project_root):
            print(f"无效的项目目录: {project_root}")
            return

        java_files = self._collect_java_files(project_root)

        # 第一遍：收集类、继承关系和方法信息
        for file_path in java_files:
            self._parse_file_info(file_path)

        # 第二遍：分析调用关系
        for file_path in java_files:
            self._analyze_call_relations(file_path)

        # 输出
        os.makedirs(output_dir, exist_ok=True)
        self._write_method_csv(os.path.join(output_dir, "method.csv"))
        self._generate_call_matrix(os.path.join(output_dir, "method_adj_matrix.csv"))
    
    def _collect_java_files(self, root_dir: str) -> List[str]:
        java_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".java"):
                    java_files.append(os.path.join(root, file))
        return java_files
    
    def _parse_file_info(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                source_code = file.read()
            
            tree = javalang.parse.parse(source_code)
            self.file_trees[file_path] = tree
            package_name = tree.package.name if tree.package else ""
            self.package_names[file_path] = package_name
            
            # 创建导入映射
            import_map: Dict[str, str] = {}
            for imp in tree.imports:
                if not imp.wildcard:
                    # import x.y.Z -> map["Z"] = "x.y.Z"
                    import_name = imp.path.split('.')[-1]
                    import_map[import_name] = imp.path
                else:
                    # 记录通配符导入包前缀（主要用于参考；精确匹配仍以类名为主）
                    import_map[imp.path.rstrip('.*')] = imp.path
            self.import_mapping[file_path] = import_map
            
            # 处理继承/实现关系
            for _, node in tree.filter(javalang.tree.ClassDeclaration):
                self._process_class_declaration(file_path, node, package_name)
            for _, node in tree.filter(javalang.tree.InterfaceDeclaration):
                self._process_interface_declaration(file_path, node, package_name)
            
            # 处理方法
            for _, node in tree.filter(javalang.tree.ClassDeclaration):
                full_class_name = f"{package_name}.{node.name}" if package_name else node.name
                self._parse_class_methods(file_path, full_class_name, node)
                # 收集字段类型信息
                self._collect_field_types(file_path, full_class_name, node, package_name, import_map)
            for _, node in tree.filter(javalang.tree.InterfaceDeclaration):
                full_class_name = f"{package_name}.{node.name}" if package_name else node.name
                self._parse_class_methods(file_path, full_class_name, node)
                
        except (javalang.parser.JavaSyntaxError, UnicodeDecodeError) as e:
            print(f"解析文件 {file_path} 时出错: {str(e)}")
    
    def _process_class_declaration(self, file_path: str, node, package_name: str):
        full_class_name = f"{package_name}.{node.name}" if package_name else node.name
        import_map = self.import_mapping.get(file_path, {})

        # 处理继承
        if node.extends:
            base_class = self._resolve_class_name(node.extends.name, package_name, import_map)
            if base_class:
                self.class_hierarchy[full_class_name].append(base_class)
        
        # 处理实现
        if node.implements:
            for impl in node.implements:
                interface_name = self._resolve_class_name(impl.name, package_name, import_map)
                if interface_name:
                    self.class_hierarchy[full_class_name].append(interface_name)
    
    def _process_interface_declaration(self, file_path: str, node, package_name: str):
        full_interface_name = f"{package_name}.{node.name}" if package_name else node.name
        import_map = self.import_mapping.get(file_path, {})
        
        # 处理接口继承
        if node.extends:
            for ext in node.extends:
                base_interface = self._resolve_class_name(ext.name, package_name, import_map)
                if base_interface:
                    self.class_hierarchy[full_interface_name].append(base_interface)
    
    def _parse_class_methods(self, file_path: str, class_name: str, class_node):
        # 处理类/接口中的所有方法
        for _, node in class_node.filter(javalang.tree.MethodDeclaration):
            self._record_method(file_path, class_name, node)
        # 处理构造函数
        for _, node in class_node.filter(javalang.tree.ConstructorDeclaration):
            self._record_method(file_path, class_name, node)
        # 处理内部类（只处理直接声明的内部类，避免无限递归）
        # 直接访问类的body成员，而不是使用filter递归查找
        if hasattr(class_node, 'body') and class_node.body:
            for member in class_node.body:
                if isinstance(member, javalang.tree.ClassDeclaration):
                    inner_class_name = f"{class_name}${member.name}"
                    self._parse_class_methods(file_path, inner_class_name, member)
    
    def _record_method(self, file_path: str, class_name: str, method_node):
        # 方法名（构造函数统一为 <init>）
        method_name = method_node.name
        if isinstance(method_node, javalang.tree.ConstructorDeclaration):
            method_name = "<init>"
        
        # 参数列表：类型 + 名称
        params = []
        for param in method_node.parameters:
            param_type = self._get_type_name(param.type)
            param_name = getattr(param, 'name', '')
            # 统一格式：Type name；若缺失名称则仅保留类型
            if param_name:
                params.append(f"{param_type} {param_name}")
            else:
                params.append(f"{param_type}")
        param_str = ", ".join(params)
        
        # 签名（包含参数名）
        method_signature = f"{class_name}.{method_name}( {param_str} )"
        
        # 位置信息
        start_line = method_node.position.line if method_node.position else 0
        end_line = self._get_method_end_line(method_node, file_path)
        
        # 源码
        method_code = self._extract_method_code(file_path, start_line, end_line)
        
        # ID 分配与存储
        method_id = self.current_id
        self.current_id += 1
        
        # 存储返回类型
        if isinstance(method_node, javalang.tree.MethodDeclaration):
            return_type = self._get_type_name(method_node.return_type) if method_node.return_type else "void"
            self.method_return_types[method_id] = return_type
        else:
            self.method_return_types[method_id] = class_name  # 构造函数返回所属类
        
        self.methods[method_id] = (method_signature, method_code, class_name, file_path, start_line, end_line)
        self.class_methods[class_name].add(method_id)
        self.signature_to_id[method_signature].append(method_id)
        
        # 收集局部变量类型信息
        self._collect_local_variables(method_id, method_node, class_name)
    
    def _collect_field_types(self, file_path: str, class_name: str, class_node, package_name: str, import_map: Dict[str, str]):
        """收集类的字段类型信息"""
        for _, field_node in class_node.filter(javalang.tree.FieldDeclaration):
            field_type = self._get_type_name(field_node.type)
            # 解析为完全限定名
            resolved_type = self._resolve_class_name(field_type, package_name, import_map) or field_type
            for declarator in field_node.declarators:
                field_name = declarator.name
                self.field_types[class_name][field_name] = resolved_type
    
    def _collect_local_variables(self, method_id: int, method_node, class_name: str):
        """收集方法的局部变量类型信息"""
        # 收集方法参数
        for param in method_node.parameters:
            param_type = self._get_type_name(param.type)
            param_name = getattr(param, 'name', '')
            if param_name:
                self.local_vars[method_id][param_name] = param_type
        
        # 收集方法体中的局部变量声明（简化处理）
        if hasattr(method_node, 'body') and method_node.body:
            self._extract_local_vars_from_body(method_id, method_node.body, class_name)
    
    def _extract_local_vars_from_body(self, method_id: int, body, class_name: str):
        """从方法体中提取局部变量声明"""
        if isinstance(body, javalang.tree.BlockStatement):
            for stmt in body.statements or []:
                if isinstance(stmt, javalang.tree.LocalVariableDeclaration):
                    var_type = self._get_type_name(stmt.type)
                    for declarator in stmt.declarators:
                        var_name = declarator.name
                        self.local_vars[method_id][var_name] = var_type

    def _get_method_end_line(self, method_node, file_path: str) -> int:
        
        start_line = method_node.position.line if method_node.position else 0
        if start_line <= 0:
            return 0

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except Exception:
            return start_line  # 兜底

        n = len(lines)
        i = start_line - 1  # 0-based

        brace_depth = 0
        in_single_line_comment = False
        in_block_comment = False

        def iterate_chars(line):
            nonlocal in_single_line_comment, in_block_comment
            in_string = False
            in_char = False
            escape = False
            j = 0
            L = len(line)
            while j < L:
                ch = line[j]
                nxt = line[j+1] if j+1 < L else ''

                # 行注释
                if not in_string and not in_char and not in_block_comment and ch == '/' and nxt == '/':
                    in_single_line_comment = True
                    break
                # 块注释开始
                if not in_string and not in_char and not in_block_comment and ch == '/' and nxt == '*':
                    in_block_comment = True
                    j += 2
                    continue
                # 块注释结束
                if in_block_comment and ch == '*' and nxt == '/':
                    in_block_comment = False
                    j += 2
                    continue

                if in_block_comment:
                    j += 1
                    continue

                # 字符串
                if not in_char and ch == '"' and not in_string:
                    in_string = True
                    escape = False
                    j += 1
                    continue
                elif in_string:
                    if ch == '\\' and not escape:
                        escape = True
                    elif ch == '"' and not escape:
                        in_string = False
                    else:
                        escape = False
                    j += 1
                    continue

                # 字符字面量
                if not in_string and ch == "'" and not in_char:
                    in_char = True
                    escape = False
                    j += 1
                    continue
                elif in_char:
                    if ch == '\\' and not escape:
                        escape = True
                    elif ch == "'" and not escape:
                        in_char = False
                    else:
                        escape = False
                    j += 1
                    continue

                # 正常字符
                yield ch
                j += 1

        # 第一阶段：查找 '{' 或 ';'
        found_open_brace = False
        while i < n:
            line = lines[i]
            in_single_line_comment = False
            for ch in iterate_chars(line):
                if in_single_line_comment:
                    break
                if ch == '{':
                    found_open_brace = True
                    brace_depth = 1
                    break
                if ch == ';':
                    # 无方法体声明，以此行为结束
                    return i + 1
            if found_open_brace:
                break
            i += 1

        if not found_open_brace:
            # 没找到 '{' 也没找到 ';'，兜底
            return start_line

        # 第二阶段：括号平衡，直到归零
        while i < n:
            line = lines[i]
            in_single_line_comment = False
            for ch in iterate_chars(line):
                if in_single_line_comment:
                    break
                if ch == '{':
                    brace_depth += 1
                elif ch == '}':
                    brace_depth -= 1
                    if brace_depth == 0:
                        return i + 1
            i += 1

        # 文件结束仍未归零，兜底返回最后一行
        return n
    
    def _extract_method_code(self, file_path: str, start_line: int, end_line: int) -> str:
        """按行号提取方法代码；若 end_line < start_line，则只取 start_line 那一行避免空串。"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                lines = file.readlines()
            if start_line <= 0:
                return ""
            if end_line <= 0 or end_line < start_line:
                end_line = start_line
            start_idx = max(0, min(start_line - 1, len(lines) - 1))
            end_idx = max(start_idx, min(end_line - 1, len(lines) - 1))
            return ''.join(lines[start_idx:end_idx + 1])
        except Exception as e:
            print(f"提取方法代码出错: {str(e)}")
            return ""
    
    def _analyze_call_relations(self, file_path: str):
        """递归遍历AST并分析调用关系"""
        try:
            tree = self.file_trees[file_path]
            package_name = self.package_names[file_path]
            import_map = self.import_mapping[file_path]
            
            # 递归遍历所有节点
            self._traverse_and_analyze(tree, file_path, package_name, import_map)
            
        except Exception as e:
            print(f"分析调用关系时出错: {file_path} - {str(e)}")
    
    def _traverse_and_analyze(self, node, file_path: str, package_name: str, import_map: Dict[str, str]):
        """递归遍历AST节点并分析调用关系"""
        # 更新上下文
        if hasattr(node, 'position') and node.position:
            self._update_context(file_path, node.position)
        
        # 处理方法调用（不使用elif，因为需要递归处理子节点）
        if isinstance(node, javalang.tree.MethodInvocation):
            self._process_method_invocation(file_path, node, package_name, import_map)
        
        # 处理构造函数调用
        if isinstance(node, javalang.tree.ClassCreator):
            self._process_constructor_call(file_path, node, package_name, import_map)
        
        # 处理方法引用
        if isinstance(node, javalang.tree.MethodReference):
            self._process_method_reference(file_path, node, package_name, import_map)
        
        # 处理显式构造函数调用（super()/this()）
        if isinstance(node, javalang.tree.ExplicitConstructorInvocation):
            self._process_explicit_constructor(file_path, node, package_name, import_map)
        
        # Lambda 表达式 - 递归处理其体
        if isinstance(node, javalang.tree.LambdaExpression):
            self._process_lambda_expression(file_path, node, package_name, import_map)
        
        # 递归遍历子节点（无论是否匹配到特定节点类型，都要遍历子节点）
        for child in self._get_children(node):
            if child is not None:
                self._traverse_and_analyze(child, file_path, package_name, import_map)
    
    def _get_children(self, node):
        """获取AST节点的所有子节点"""
        children = []
        if hasattr(node, '__dict__'):
            for key, value in node.__dict__.items():
                if key == 'position' or key.startswith('_'):
                    continue
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, javalang.tree.Node):
                            children.append(item)
                elif isinstance(value, javalang.tree.Node):
                    children.append(value)
        return children
    
    def _update_context(self, file_path: str, position):
        """根据位置更新当前解析上下文（支持嵌套方法）"""
        if not position:
            return
        
        line = position.line
        
        # 查找所有包含当前行的方法（可能有嵌套）
        matching_methods = []
        for method_id, (_, _, _, m_file_path, start, end) in self.methods.items():
            if m_file_path == file_path and start <= line <= end:
                matching_methods.append((method_id, start, end))
        
        if not matching_methods:
            self.current_context = []
            return
        
        # 按嵌套深度排序（最内层的方法在最前面）
        matching_methods.sort(key=lambda x: (x[2] - x[1], -x[1]), reverse=True)
        self.current_context = [m[0] for m in matching_methods]
    
    def _process_method_invocation(self, file_path: str, node, package_name: str, import_map: Dict[str, str]):
        if not self.current_context:
            return
        
        caller_id = self.current_context[0]
        method_name = node.member
        
        # 解析目标类
        target_class = None
        if node.qualifier:
            qualifier = node.qualifier
            if isinstance(qualifier, str):
                if qualifier == "this":
                    target_class = self.methods[caller_id][2]
                elif qualifier == "super":
                    # super.method() 调用父类方法
                    caller_class = self.methods[caller_id][2]
                    if caller_class in self.class_hierarchy:
                        # 取第一个父类（通常只有一个父类）
                        parents = self.class_hierarchy[caller_class]
                        if parents:
                            target_class = parents[0]
                else:
                    target_class = self._resolve_class_name(qualifier, package_name, import_map)
                    # 可能是字段访问，尝试推断字段类型
                    if not target_class:
                        target_class = self._resolve_field_type(caller_id, qualifier)
            elif isinstance(qualifier, javalang.tree.MemberReference):
                # 处理链式调用：obj.field.method()
                expr_type = self._infer_expression_type(qualifier, caller_id, package_name, import_map)
                if expr_type:
                    target_class = self._resolve_class_name(expr_type, package_name, import_map)
            elif isinstance(qualifier, javalang.tree.MethodInvocation):
                # 处理链式方法调用：method1().method2()
                expr_type = self._infer_expression_type(qualifier, caller_id, package_name, import_map)
                if expr_type:
                    target_class = self._resolve_class_name(expr_type, package_name, import_map)
            elif hasattr(qualifier, 'name'):
                target_class = self._resolve_class_name(qualifier.name, package_name, import_map)
            elif hasattr(qualifier, 'type'):
                target_class = self._resolve_class_name(self._get_type_name(qualifier.type), package_name, import_map)
            else:
                # 尝试推断表达式类型
                expr_type = self._infer_expression_type(qualifier, caller_id, package_name, import_map)
                if expr_type:
                    target_class = self._resolve_class_name(expr_type, package_name, import_map)
        else:
            # 无限定符：可能是实例方法（this.method()）或静态方法
            # 首先检查是否是静态导入
            target_class = self._resolve_static_import(file_path, method_name, import_map)
            if not target_class:
                # 使用上下文类
                target_class = self._find_context_class(file_path, node.position)
        
        # 获取参数类型
        arg_types: List[Optional[str]] = []
        for arg in node.arguments:
            arg_type = self._infer_expression_type(arg, caller_id, package_name, import_map)
            if arg_type == "this":
                arg_type = self.methods[caller_id][2]
            if arg_type:
                arg_types.append(arg_type)
        
        callee_ids = []
        
        # 如果找到目标类，尝试精确匹配
        if target_class:
            callee_ids = self._find_matching_methods(target_class, method_name, arg_types)
        
        # 如果找不到精确匹配，尝试模糊匹配
        if not callee_ids:
            # 首先尝试在当前类中查找同名方法
            caller_class = self.methods[caller_id][2]
            for method_id in self.class_methods.get(caller_class, []):
                signature = self.methods[method_id][0]
                # 检查方法名是否匹配（考虑参数数量）
                if self._simple_method_match(signature, method_name, len(arg_types)):
                    if method_id not in callee_ids:
                        callee_ids.append(method_id)
            
            # 在继承链中查找
            if caller_class in self.class_hierarchy:
                for parent in self.class_hierarchy[caller_class]:
                    for method_id in self.class_methods.get(parent, []):
                        signature = self.methods[method_id][0]
                        if self._simple_method_match(signature, method_name, len(arg_types)):
                            if method_id not in callee_ids:
                                callee_ids.append(method_id)
            
            # 如果还是找不到，在已知的所有类中查找同名方法（放宽匹配）
            if not callee_ids:
                for class_name in self.class_methods:
                    for method_id in self.class_methods[class_name]:
                        signature = self.methods[method_id][0]
                        # 简单匹配：方法名在签名中且参数数量匹配
                        if self._simple_method_match(signature, method_name, len(arg_types)):
                            if method_id not in callee_ids:
                                callee_ids.append(method_id)
        
        # 记录调用关系
        for callee_id in callee_ids:
            if callee_id != caller_id:  # 避免自己调用自己（除非是递归）
                self.call_relations[caller_id].add(callee_id)
    
    def _process_constructor_call(self, file_path: str, node, package_name: str, import_map: Dict[str, str]):
        if not self.current_context:
            return
        
        caller_id = self.current_context[0]
        
        # 解析构造函数所属的类
        class_type = node.type
        class_name = self._get_type_name(class_type)
        target_class = self._resolve_class_name(class_name, package_name, import_map)
        
        if not target_class:
            return
        
        # 获取参数类型
        arg_types: List[Optional[str]] = []
        if node.arguments:
            for arg in node.arguments:
                t = self._infer_expression_type(arg, caller_id, package_name, import_map)
                if t == "this":
                    t = self.methods[caller_id][2]
                if t:
                    arg_types.append(t)
        
        # 查找构造函数
        callee_ids = self._find_matching_methods(target_class, "<init>", arg_types)
        
        # 记录调用关系
        for callee_id in callee_ids:
            self.call_relations[caller_id].add(callee_id)
    
    def _process_method_reference(self, file_path: str, node, package_name: str, import_map: Dict[str, str]):
        if not self.current_context:
            return
        
        caller_id = self.current_context[0]
        method_name = node.method
        
        # 解析方法所属的类
        target_class = None
        if node.expression:
            expr_type = self._infer_expression_type(node.expression, caller_id, package_name, import_map)
            if expr_type == "this":
                expr_type = self.methods[caller_id][2]
            if expr_type:
                target_class = self._resolve_class_name(expr_type, package_name, import_map)
        
        if not target_class:
            return
        
        # 查找匹配的方法（方法引用通常不传递参数，但需要匹配方法签名）
        callee_ids = self._find_matching_methods(target_class, method_name, [])
        
        # 记录调用关系
        for callee_id in callee_ids:
            self.call_relations[caller_id].add(callee_id)
    
    def _process_lambda_expression(self, file_path: str, node, package_name: str, import_map: Dict[str, str]):
        """处理 Lambda 表达式中的方法调用"""
        # Lambda 表达式内的代码应该在外部上下文中处理
        # 这里不做特殊处理，让递归遍历处理
        pass
    
    def _process_explicit_constructor(self, file_path: str, node, package_name: str, import_map: Dict[str, str]):
        """处理显式构造函数调用（如 super() 或 this()）"""
        if not self.current_context:
            return
        
        caller_id = self.current_context[0]
        _, _, class_name, _, _, _ = self.methods[caller_id]
        
        if node.type == 'super':
            if class_name in self.class_hierarchy:
                parent_classes = self.class_hierarchy[class_name]
                for parent in parent_classes:
                    arg_types = []
                    for arg in node.arguments or []:
                        t = self._infer_expression_type(arg, caller_id, package_name, import_map)
                        if t == "this":
                            t = class_name
                        if t:
                            arg_types.append(t)
                    callee_ids = self._find_matching_methods(parent, "<init>", arg_types)
                    for callee_id in callee_ids:
                        self.call_relations[caller_id].add(callee_id)
        else:
            # this() 调用同一个类的其他构造函数
            arg_types = []
            for arg in node.arguments or []:
                t = self._infer_expression_type(arg, caller_id, package_name, import_map)
                if t == "this":
                    t = class_name
                if t:
                    arg_types.append(t)
            callee_ids = self._find_matching_methods(class_name, "<init>", arg_types)
            for callee_id in callee_ids:
                # 避免自己调用自己
                if callee_id != caller_id:
                    self.call_relations[caller_id].add(callee_id)
    
    def _process_chained_call(self, file_path: str, node, package_name: str, import_map: Dict[str, str]):
        """处理链式方法调用（非常简化的推断）"""
        if not self.current_context:
            return
        
        caller_id = self.current_context[0]
        
        # 递归处理链式调用的每个部分
        chain: List[str] = []
        current = node
        while current and isinstance(current, javalang.tree.MemberReference):
            chain.insert(0, current.member)
            current = getattr(current, 'qualifier', None) if hasattr(current, 'qualifier') else None
        
        # 解析链的起点
        target_class = None
        if isinstance(current, javalang.tree.MemberReference):
            if current.member == "this":
                target_class = self.methods[caller_id][2]
            else:
                target_class = self._resolve_class_name(current.member, package_name, import_map)
        elif hasattr(current, 'type'):
            target_class = self._resolve_class_name(self._get_type_name(current.type), package_name, import_map)
        
        if not target_class:
            return
        
        # 处理链中的每个方法调用
        for method_name in chain:
            callee_ids = self._find_matching_methods(target_class, method_name, [])
            if callee_ids:
                return_type = self._get_method_return_type(callee_ids[0])
                if return_type:
                    target_class = self._resolve_class_name(return_type, package_name, import_map) or target_class
                for callee_id in callee_ids:
                    self.call_relations[caller_id].add(callee_id)
    
    def _get_method_return_type(self, method_id: int) -> Optional[str]:
        """获取方法的返回类型"""
        return self.method_return_types.get(method_id)
    
    def _infer_expression_type(self, expr, caller_id: Optional[int] = None, package_name: str = "", import_map: Dict[str, str] = None) -> Optional[str]:
        """推断表达式类型（改进版）"""
        if import_map is None:
            import_map = {}
        
        # 字面量
        if isinstance(expr, javalang.tree.Literal):
            if "'" in str(expr.value) or '"' in str(expr.value):
                return "String"
            elif '.' in str(expr.value) and not expr.value.startswith('.'):
                return "double"
            elif str(expr.value).replace('-', '').replace('+', '').isdigit():
                return "int"
            elif str(expr.value) in ['true', 'false']:
                return "boolean"
            elif str(expr.value).endswith('L') or str(expr.value).endswith('l'):
                return "long"
            elif str(expr.value).endswith('F') or str(expr.value).endswith('f'):
                return "float"
            elif str(expr.value).endswith('D') or str(expr.value).endswith('d'):
                return "double"
        
        # this 引用
        elif isinstance(expr, javalang.tree.This):
            return "this"
        
        # super 引用
        elif isinstance(expr, javalang.tree.SuperMemberReference):
            if caller_id:
                caller_class = self.methods[caller_id][2]
                if caller_class in self.class_hierarchy:
                    parents = self.class_hierarchy[caller_class]
                    if parents:
                        return parents[0]
            return None
        
        # 字段访问
        elif isinstance(expr, javalang.tree.MemberReference):
            if caller_id and hasattr(expr, 'member'):
                # 尝试从局部变量或字段中查找
                var_type = self.local_vars[caller_id].get(expr.member)
                if var_type:
                    return var_type
                caller_class = self.methods[caller_id][2]
                field_type = self.field_types[caller_class].get(expr.member)
                if field_type:
                    return field_type
                # 可能是方法调用的结果，尝试推断
                return expr.member
        
        # 方法调用
        elif isinstance(expr, javalang.tree.MethodInvocation):
            return self._infer_method_return_type(expr, caller_id, package_name, import_map)
        
        # 构造函数调用
        elif isinstance(expr, javalang.tree.ClassCreator):
            class_type = expr.type
            class_name = self._get_type_name(class_type)
            return self._resolve_class_name(class_name, package_name, import_map)
        
        # 数组访问
        elif isinstance(expr, javalang.tree.ArraySelector):
            if hasattr(expr, 'postfix_operators'):
                # 数组类型，获取元素类型
                if hasattr(expr, 'primary'):
                    primary_type = self._infer_expression_type(expr.primary, caller_id, package_name, import_map)
                    if primary_type and primary_type.endswith('[]'):
                        return primary_type[:-2]
                    return primary_type
        
        # 类型转换
        elif isinstance(expr, javalang.tree.Cast):
            if hasattr(expr, 'type'):
                return self._get_type_name(expr.type)
        
        # 有类型信息的节点
        elif hasattr(expr, 'type'):
            return self._get_type_name(expr.type)
        
        # 三元运算符
        elif isinstance(expr, javalang.tree.TernaryExpression):
            # 返回 then 表达式的类型
            if hasattr(expr, 'if_true'):
                return self._infer_expression_type(expr.if_true, caller_id, package_name, import_map)
        
        return None
    
    def _infer_method_return_type(self, method_invocation, caller_id: Optional[int] = None, package_name: str = "", import_map: Dict[str, str] = None) -> Optional[str]:
        """推断方法调用的返回类型（改进版）"""
        if import_map is None:
            import_map = {}
        
        method_name = method_invocation.member
        
        # 解析目标类
        target_class = None
        if method_invocation.qualifier:
            qualifier = method_invocation.qualifier
            if isinstance(qualifier, str):
                if qualifier == "this" and caller_id:
                    target_class = self.methods[caller_id][2]
                else:
                    target_class = self._resolve_class_name(qualifier, package_name, import_map)
            else:
                expr_type = self._infer_expression_type(qualifier, caller_id, package_name, import_map)
                if expr_type:
                    target_class = self._resolve_class_name(expr_type, package_name, import_map)
        else:
            if caller_id:
                target_class = self.methods[caller_id][2]
        
        if target_class:
            # 查找方法并获取返回类型
            arg_types = []
            for arg in method_invocation.arguments:
                arg_type = self._infer_expression_type(arg, caller_id, package_name, import_map)
                if arg_type:
                    arg_types.append(arg_type)
            
            method_ids = self._find_matching_methods(target_class, method_name, arg_types)
            if method_ids:
                # 返回第一个匹配方法的返回类型
                return_type = self.method_return_types.get(method_ids[0])
                if return_type:
                    return return_type
        
        # 默认返回类型推断
        if method_invocation.qualifier:
            qualifier = method_invocation.qualifier
            if isinstance(qualifier, str):
                return qualifier
            elif hasattr(qualifier, 'name'):
                return qualifier.name
        
        return None
    
    def _resolve_field_type(self, caller_id: int, field_name: str) -> Optional[str]:
        """解析字段类型"""
        if caller_id in self.methods:
            caller_class = self.methods[caller_id][2]
            # 在当前类中查找
            if caller_class in self.field_types:
                field_type = self.field_types[caller_class].get(field_name)
                if field_type:
                    return field_type
            # 在父类中递归查找
            if caller_class in self.class_hierarchy:
                for parent in self.class_hierarchy[caller_class]:
                    if parent in self.field_types:
                        field_type = self.field_types[parent].get(field_name)
                        if field_type:
                            return field_type
        return None
    
    def _resolve_static_import(self, file_path: str, method_name: str, import_map: Dict[str, str]) -> Optional[str]:
        """解析静态导入的方法"""
        # 检查是否是静态导入的方法
        # 简化处理：在已知类中查找静态方法
        for class_name in self.class_methods:
            for method_id in self.class_methods[class_name]:
                signature = self.methods[method_id][0]
                # 检查是否是静态方法（简化：通过方法名匹配）
                if method_name in signature:
                    # 检查是否是导入的类
                    short_name = class_name.split('.')[-1]
                    if short_name in import_map and import_map[short_name] == class_name:
                        return class_name
        return None
    
    def _resolve_class_name(self, name: str, package_name: str, import_map: Dict[str, str]) -> Optional[str]:
        """解析类名到完全限定名（基于当前文件 import_map 与已知类集合）"""
        if '.' in name and name in self.class_methods:
            return name
        
        if name in import_map:
            return import_map[name]
        
        full_name = f"{package_name}.{name}" if package_name else name
        if full_name in self.class_methods:
            return full_name
        
        java_lang_name = f"java.lang.{name}"
        if java_lang_name in self.class_methods:
            return java_lang_name
        
        for cls in self.class_methods:
            if cls.endswith(f".{name}"):
                return cls
        
        return None
    
    def _find_context_class(self, file_path: str, position) -> Optional[str]:
        """根据位置查找上下文类（使用当前行所在的方法的类名）"""
        if not position:
            return None
        for _, (_, _, class_name, _, start, end) in self.methods.items():
            if start <= position.line <= end:
                return class_name
        return None
    
    def _find_matching_methods(self, class_name: str, method_name: str, arg_types: List[str]) -> List[int]:
        """查找匹配的方法，递归查找继承链和接口（改进版）"""
        method_ids: List[int] = []
        visited_classes = set()
        
        def find_recursive(cls_name: str):
            if cls_name in visited_classes:
                return
            visited_classes.add(cls_name)
            
            # 查找当前类的方法
            if cls_name in self.class_methods:
                for method_id in self.class_methods[cls_name]:
                    signature = self.methods[method_id][0]
                    if self._match_signature(signature, method_name, arg_types):
                        method_ids.append(method_id)
            
            # 递归查找父类
            if cls_name in self.class_hierarchy:
                for parent in self.class_hierarchy[cls_name]:
                    find_recursive(parent)
        
        find_recursive(class_name)
        return method_ids
    
    def _simple_method_match(self, signature: str, method_name: str, arg_count: int) -> bool:
        """简单的.method名匹配（不考虑类型）"""
        # 检查方法名是否在签名中
        pattern = f".{method_name}("
        if pattern not in signature:
            # 构造函数特殊处理
            if method_name == "<init>" and "<init>(" in signature:
                pattern = "<init>("
            else:
                return False
        
        # 提取参数部分（从最后一个(开始，避免匹配泛型中的括号）
        last_paren = signature.rfind('(')
        if last_paren == -1:
            return arg_count == 0
        
        params_str = signature[last_paren + 1:].rstrip(')').strip()
        if not params_str:
            return arg_count == 0
        
        # 计算参数数量（通过逗号分隔，但需要考虑泛型和嵌套括号）
        # 简化处理：按逗号分隔，过滤掉空字符串
        param_list = [p.strip() for p in params_str.split(',') if p.strip()]
        return len(param_list) == arg_count
    
    def _match_signature(self, signature: str, method_name: str, arg_types: List[str]) -> bool:
        """检查方法签名是否匹配（改进版，支持类型兼容性匹配）"""
        match = re.match(r".*?\.(\w+)\((.*?)\)", signature)
        if not match:
            return False
        
        sig_method_name = match.group(1)
        sig_params = match.group(2)
        
        if sig_method_name != method_name and not (sig_method_name == "<init>" and method_name == "<init>"):
            return False
        
        # 允许签名里出现 "Type name"；只取类型（第一个空格前或泛型整体）来对比
        def extract_type(p: str) -> str:
            p = p.strip()
            if not p:
                return ""
            # 处理形如 "List<String> ids" 或 "Map<K, V> m"
            # 提取泛型之前的部分
            parts = p.split()
            if len(parts) > 0:
                return parts[0]
            return p
        
        sig_param_list_raw = [p.strip() for p in sig_params.split(',')] if sig_params else []
        sig_param_types = [extract_type(p) for p in sig_param_list_raw if p != ""]
        
        # 如果参数数量不匹配，返回False
        if len(sig_param_types) != len(arg_types):
            return False
        
        # 如果参数列表为空，直接匹配
        if len(sig_param_types) == 0:
            return True
        
        # 类型兼容性检查
        primitive_map = {
            "int": "Integer",
            "long": "Long",
            "double": "Double",
            "float": "Float",
            "boolean": "Boolean",
            "char": "Character",
            "byte": "Byte",
            "short": "Short"
        }
        
        for i, param_type in enumerate(arg_types):
            # 如果参数类型未知，跳过检查（允许匹配）
            if not param_type:
                continue
            
            sig_t = sig_param_types[i]
            if not sig_t:
                return False
            
            # 完全匹配
            if sig_t == param_type:
                continue
            
            # 原始类型和包装类型匹配
            if param_type in primitive_map and sig_t == primitive_map[param_type]:
                continue
            if sig_t in primitive_map and primitive_map[sig_t] == param_type:
                continue
            
            # 包名匹配（允许不同包名的同名类型，但这里简化处理）
            sig_base = sig_t.split('.')[-1] if '.' in sig_t else sig_t
            param_base = param_type.split('.')[-1] if '.' in param_type else param_type
            if sig_base == param_base:
                continue
            
            # 完全限定名部分匹配
            if sig_t.endswith(f".{param_base}") or param_type.endswith(f".{sig_base}"):
                continue
            
            # 泛型类型简化匹配（忽略泛型参数）
            sig_generic = sig_t.split('<')[0] if '<' in sig_t else sig_t
            param_generic = param_type.split('<')[0] if '<' in param_type else param_type
            if sig_generic == param_generic:
                continue
            
            return False
        
        return True
    
    def _get_type_name(self, type_node) -> str:
        """获取类型名称，处理泛型（仅保留主类型名与泛型参数的主类型名）"""
        if isinstance(type_node, javalang.tree.ReferenceType):
            base_name = '.'.join(type_node.name) if isinstance(type_node.name, list) else type_node.name
            if type_node.arguments:
                arg_names = []
                for arg in type_node.arguments:
                    if hasattr(arg, 'type') and arg.type is not None:
                        arg_names.append(self._get_type_name(arg.type))
                    else:
                        arg_names.append("?")
                return f"{base_name}<{', '.join(arg_names)}>"
            return base_name
        elif hasattr(type_node, 'name'):
            return type_node.name
        return ""
    
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print
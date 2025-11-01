import os
import csv
import javalang
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class JavaImportAnalyzer:
    def __init__(self):
        self.class_to_file: Dict[str, str] = {}
        self.package_classes: Dict[str, Set[str]] = defaultdict(set)
        self.all_classes: List[str] = []
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
    
    def analyze_project(self, project_root: str, output_file: str = "file_adj_matrix.csv"):
        # 验证项目路径
        if not os.path.isdir(project_root):
            print(f"无效的项目目录: {project_root}")
            return
        
        # 收集所有Java文件
        java_files = self._collect_java_files(project_root)
        
        # 第一遍解析：收集类信息
        for file_path in java_files:
            self._parse_class_info(file_path)
        
        # 第二遍解析：分析依赖关系
        for file_path in java_files:
            self._analyze_file_dependencies(file_path)
        
        # 生成邻接矩阵并写入CSV
        self._generate_adjacency_matrix(output_file)
    
    def _collect_java_files(self, root_dir: str) -> List[str]:
        java_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".java"):
                    java_files.append(os.path.join(root, file))
        return java_files
    
    def _parse_class_info(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                source_code = file.read()
            
            tree = javalang.parse.parse(source_code)
            package_name = tree.package.name if tree.package else ""
            
            # 处理文件中的所有类声明
            for path, node in tree.filter(javalang.tree.ClassDeclaration):
                full_class_name = f"{package_name}.{node.name}" if package_name else node.name
                self.all_classes.append(full_class_name)
                self.class_to_file[full_class_name] = file_path
                self.package_classes[package_name].add(full_class_name)
            
            # 处理接口声明
            for path, node in tree.filter(javalang.tree.InterfaceDeclaration):
                full_class_name = f"{package_name}.{node.name}" if package_name else node.name
                self.all_classes.append(full_class_name)
                self.class_to_file[full_class_name] = file_path
                self.package_classes[package_name].add(full_class_name)
                
        except (javalang.parser.JavaSyntaxError, UnicodeDecodeError) as e:
            print(f"解析文件 {file_path} 时出错: {str(e)}")
    
    def _analyze_file_dependencies(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                source_code = file.read()
            
            tree = javalang.parse.parse(source_code)
            package_name = tree.package.name if tree.package else ""
            
            # 获取当前文件的主类
            main_class = None
            for path, node in tree.filter((javalang.tree.ClassDeclaration, javalang.tree.InterfaceDeclaration)):
                if not main_class:
                    main_class = f"{package_name}.{node.name}" if package_name else node.name
            
            if not main_class:
                return
            
            # 处理导入依赖
            for imp in tree.imports:
                imported = imp.path
                
                # 处理通配符导入
                if imp.wildcard:
                    imported_pkg = imported.rstrip('.*')
                    # 添加该包下的所有本地类
                    for cls in self.package_classes.get(imported_pkg, []):
                        self.dependencies[main_class].add(cls)
                # 处理单类导入
                else:
                    if imported in self.all_classes:
                        self.dependencies[main_class].add(imported)
            
            # 处理同包下的隐式导入
            if package_name in self.package_classes:
                for cls in self.package_classes[package_name]:
                    if cls != main_class:
                        self.dependencies[main_class].add(cls)
            
            # 处理完全限定名引用
            for _, node in tree.filter(javalang.tree.ReferenceType):
                # 尝试解析完全限定名
                if hasattr(node, 'name') and '.' in node.name:
                    possible_class = node.name
                    if possible_class in self.all_classes:
                        self.dependencies[main_class].add(possible_class)
            
        except (javalang.parser.JavaSyntaxError, UnicodeDecodeError) as e:
            print(f"分析依赖 {file_path} 时出错: {str(e)}")
    
    def _generate_adjacency_matrix(self, output_dir: str):
        if not self.all_classes:
            print("未找到可分析的类")
            return
        
        # 排序类名以确保顺序一致
        sorted_classes = sorted(set(self.all_classes))
        class_index = {cls: idx for idx, cls in enumerate(sorted_classes)}
        size = len(sorted_classes)
        
        # 初始化邻接矩阵
        adj_matrix = [[0] * size for _ in range(size)]
        
        # 填充依赖关系
        for src_class, deps in self.dependencies.items():
            if src_class not in class_index:
                continue
            src_idx = class_index[src_class]
            for dep_class in deps:
                if dep_class in class_index:
                    dep_idx = class_index[dep_class]
                    adj_matrix[src_idx][dep_idx] = 1
        
        # 写入CSV文件
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "file_adj_matrix.csv")
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入标题行
            writer.writerow([''] + sorted_classes)
            
            # 写入矩阵行
            for i, cls in enumerate(sorted_classes):
                row = [cls] + adj_matrix[i]
                writer.writerow(row)
        
        print(f"邻接矩阵已写入: {output_file}")


if __name__ == "__main__":
    import sys
    
    project_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "out"

    analyzer = JavaImportAnalyzer()
    analyzer.analyze_project(project_path, output_dir)
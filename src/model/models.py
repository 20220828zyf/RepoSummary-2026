from dataclasses import dataclass, field
from typing import List

@dataclass
class Function:
    func_id: int 
    func_name: str
    func_desc: str
    func_file: str
    func_flow: str
    func_notf: str 
    func_code: str
    func_fullName: str 
    func_txt_vector: List[float] = field(default_factory=list)

@dataclass
class File:
    file_id: int
    file_name: str
    file_path: str
    file_code: str = ""
    file_desc: str = ""
    file_discode: str = ""
    func_list: List[Function] = field(default_factory=list)
    file_txt_vector: List[float] = field(default_factory=list)

class file_Cluster:
    cluster_id: int
    cluster_desc: str
    cluster_file_list: List[File]

    def __init__(self, cluster_id, cluster_desc, cluster_file_list):
        self.cluster_id = cluster_id
        self.cluster_desc = cluster_desc
        self.cluster_file_list = cluster_file_list

class method_Cluster:
    cluster_id: int
    cluster_desc: str
    cluster_func_list: List[Function]

    def __init__(self, cluster_id, cluster_desc, cluster_func_list):
        self.cluster_id = cluster_id
        self.cluster_desc = cluster_desc
        self.cluster_func_list = cluster_func_list

class Feature:
    cluster_id: int
    feature_id: int
    feature_desc: str
    feature_flow: str = ""
    feature_notf: str = ""
    feature_func_list: List[Function]

    def __init__(self, cluster_id, feature_id, feature_desc, feature_func_list):
        self.cluster_id = cluster_id
        self.feature_id = feature_id
        self.feature_desc = feature_desc
        self.feature_func_list = feature_func_list


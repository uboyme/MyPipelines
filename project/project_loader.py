# 将llm生成的sysml推荐结果 拼接为完整的magic项目，这里处理两种，一种是json，一种是xmi格式
import os
import json
from utils.namespace import NS
import xml.etree.ElementTree as ET  # 导入解析XML所需的包
from enum import Enum
# from code.src.evaluators.syntax_evaluator.syntax_validator import XMISyntaxValidator
import re

class ContentType(Enum):
    JSON = "json"
    XMI = "xmi"

    def __str__(self):
        return self.value

class ProjectLoader:
    def __init__(self, project_path: str, content_type: ContentType = ContentType.XMI):
        self.project_path = project_path  # 项目文件地址，目前只接受 导出的xmi文件的地址
        self.project_data = None
        self.content_type = content_type  # 项目内容类型，默认为XMI
        
    def wrap_project(self,new_content):
        '''
        new_content: str, 新的内容，可能是json或者xmi格式。把llm生成的sysml推荐结果拼接为完整的sysml项目
        '''
        pass
    
    def load_local_project(self):
        """
        加载XMI项目文件
        """
        if not os.path.exists(self.project_path):
            raise FileNotFoundError(f"Project file not found: {self.project_path}")
        with open(self.project_path, 'r', encoding='utf-8') as file:
           self.project_data = file.read()
    
    def save_project(self, output_path: str):
        """
        保存项目到指定路径
        """
        if not self.project_data:
            raise ValueError("No project data to save.")
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(self.project_data)
        print(f"Project saved to {output_path}")
        
        
    
 
class MagicDrawProjectLoader(ProjectLoader):
    """
    MagicDraw项目加载器，专门处理MagicDraw导出的XMI文件
    """

    def __init__(self, project_path: str,use_llm_structure: bool = False, llm=None,max_node_num=1000, NS: dict = NS):
        super().__init__(project_path)
        self.namespace = NS  # MagicDraw的命名空间
        self.project_xmi = None # 存储加载的XMI内容，是XML的模型树。最后填充完之后，文本更为project_data属性
        self.use_llm_structure = use_llm_structure  # 是否使用LLM决定要插入的元素在哪个位置目录
        self.llm=llm # LLM模型，用于辅助结构决策
        self.prompt="""
You are a system modeling assistant helping with SysML XMI model construction. Based on the provided model structure, current user modeling focus, and a new model fragment, your task is to decide the most appropriate parent element (by ID) under which the new fragment should be inserted.
You will be given:
- {model_structure}: A hierarchical summary of the current SysML model, represented as a tree. Each node includes its `xmi:id`, element type, name, and any relevant attributes.
- {current_position}: The `xmi:id` of the current element the user is working on, representing the active context.
- {incremental_elem}: A new XML fragment (SysML-compliant) that needs to be inserted into the model.

Your goal is to identify the `xmi:id` of the parent element in the existing model under which the new element should be inserted. Focus on structural and semantic consistency with the model hierarchy and user’s current context.

Only output the `xmi:id` of the recommended parent element. Do not return any explanation or other content.

Example output:
`_18_0_1_ae502ce_1421696920523_400465_47687`
        """
        self.max_node_num=1000
        # 判断，如果use_llm_structure为True，则需要传入llm模型
        if self.use_llm_structure and not self.llm:
            raise ValueError("use_llm_structure is True, but no LLM model provided.")
        self.id_element_map = {}  # 用于存储ID到XML元素的映射
        self.load_project_xmi()
        
        

    def get_attr(self,elem,prefix_colon_key):
        if ":" not in prefix_colon_key:
            return elem.attrib.get(prefix_colon_key)
        prefix,key=prefix_colon_key.split(":")
        namespace=self.namespace.get(prefix)
        if namespace:
            return elem.attrib.get(f"{{{namespace}}}{key}")
        else:
            return elem.attrib.get(prefix_colon_key)
        
    def init_id_element_map(self):
        """
        初始化ID到XML元素的映射
        """
        if self.project_xmi is None:
            raise ValueError("Project XMI is not loaded.")
        for elem in self.project_xmi.iter():
            elem_id = self.get_attr(elem, 'xmi:id')
            if elem_id:
                self.id_element_map[elem_id] = elem

    def to_string(self, element, pretty=False, xml_declaration=True):
        """
        将XML Element转换为字符串格式

        参数:
        - element: 要序列化的XML元素
        - pretty: 是否美化缩进（默认False）
        - xml_declaration: 是否添加XML声明头（默认True）

        返回:
        - XML字符串
        """
        import xml.etree.ElementTree as ET
        from xml.dom import minidom

        # 注册命名空间
        for prefix, uri in NS.items():
            ET.register_namespace(prefix, uri)

        # 直接序列化为字节串
        rough_bytes = ET.tostring(
            element,
            encoding='utf-8',
            method='xml',
            xml_declaration=xml_declaration
        )

        # 不需要美化，直接返回字符串
        if not pretty:
            return rough_bytes.decode('utf-8')

        # 美化缩进
        reparsed = minidom.parseString(rough_bytes)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        # 如果不需要XML声明头，就移除第一行
        if not xml_declaration:
            lines = pretty_xml.split('\n')
            if lines[0].strip().startswith('<?xml'):
                lines = lines[1:]
            pretty_xml = '\n'.join(lines)
        else:
            pretty_xml = re.sub(r'^<\?xml[^>]*\?>', '<?xml version="1.0" encoding="UTF-8"?>', pretty_xml)

        # 去除空行
        pretty_xml = "\n".join([line for line in pretty_xml.split("\n") if line.strip() != ""])

        return pretty_xml


    def wrap_parent_content(self,content: str) -> str:
        return f'''
<xmi:XMI xmlns:StandardProfile="http://www.omg.org/spec/UML/20131001/StandardProfile"
         xmlns:uml="http://www.omg.org/spec/UML/20131001"
         xmlns:sysml="http://www.omg.org/spec/SysML/20181001/SysML"
         xmlns:Dependency_Matrix_Profile="http://www.magicdraw.com/schemas/Dependency_Matrix_Profile.xmi"
         xmlns:MD_Customization_for_SysML__additional_stereotypes="http://www.magicdraw.com/spec/Customization/180/SysML"
         xmlns:xmi="http://www.omg.org/spec/XMI/20131001"
         xmlns:MD_Customization_for_Requirements__additional_stereotypes="http://www.magicdraw.com/spec/Customization/180/Requirements"
         xmlns:Custom_Stereotypes="http://www.magicdraw.com/schemas/Custom_Stereotypes.xmi"
         xmlns:Stereotypes="http://www.magicdraw.com/schemas/Stereotypes.xmi">
    {content}
   </xmi:XMI>'''

    def load_xmi_string(self, xmi_string: str):
        try:
            res = ET.fromstring(xmi_string)
            return True, res
        except ET.XMLSyntaxError as e:
            print(f"XMI 解析失败: {e}")
            return False, None
    
    def get_model_and_stereotypes(self, content):
        # xmiSyntaxValidator = XMISyntaxValidator()
        wrapped_content = self.wrap_parent_content(content)
        is_valid, node_xml = self.load_xmi_string(wrapped_content)
        if not is_valid:
            return False,None, None
        model_content_list=[]
        stereo_content_list=[]

        for child in list(node_xml):
            full_tag = child.tag  # e.g., '{http://...}Block'
            if full_tag.startswith("{"):
                uri, localname = full_tag[1:].split("}")
            else:
                uri = None
                localname = full_tag

            if uri not in NS.values():
                print(f"未知的命名空间 URI: {uri}（元素: {localname}）")
                model_content_list.append(child)
            else:
                print(f"已知命名空间元素: {localname}，命名空间: {uri}")
                stereo_content_list.append(child)
        return True, model_content_list, stereo_content_list


    def load_project_xmi(self):
        """
        加载MagicDraw项目的XMI文件,将模型树更新到project_xmi属性中
        """
        self.load_local_project()
        try:
            self.project_xmi = ET.fromstring(self.project_data)
            self.init_id_element_map()  # 初始化ID到XML元素的映射
            
        except ET.ParseError as e:
            raise ValueError(f"加载XMI失败，解析错误: {e}")
    

    def _get_model_structure(self):
        """
        获取当前项目的模型结构，只关注 <uml:Model> 元素及其下的<packagedElement> 元素。
        返回文本的化的模型结构字符串，包含每个元素的节点的信息。
        """
        def get_packaged_element_recursively(result_element, element):
            """
            递归地获取 <packagedElement> 元素及其子元素，并将其添加到 result_element 中。
            """
            # 获取当前元素的ID和类型
            element_id = self.get_attr(element, 'xmi:id')
            element_type = self.get_attr(element, 'xmi:type')
            element_name = self.get_attr(element, 'name')

            # 创建一个新的元素来存储当前元素的信息
            new_element = ET.Element('packagedElement', attrib={
                'xmi:id': element_id,
                'xmi:type': element_type,
                'name': element_name
            })

            # 将新元素添加到结果元素中
            result_element.append(new_element)

            # 递归处理子元素
            for child in element:
                if self.get_attr(child, 'xmi:type') == 'uml:PackagedElement':
                    get_packaged_element_recursively(new_element, child)

        
        
        if self.project_xmi is None:
            raise ValueError("Project XMI is not loaded.")
        
        # 找到 uml:Model 标签的元素，这个元素只有一个
        uml_model = self.project_xmi.find('.//uml:Model', namespaces=self.namespace)
        if uml_model is not None:
            temp_root = ET.Element(f"{self.namespace['uml']}Model", attrib=uml_model.attrib)
            # 递归 迭代的复制所有的保持原结构的packagedElement元素
            for child in uml_model:
                if child.tag.endswith('PackagedElement'):
                    get_packaged_element_recursively(temp_root, child)
            return self.to_string(temp_root, pretty=True, xml_declaration=False)
        else:
            raise ValueError("未找到 uml:Model 元素，请检查XMI文件是否正确。")

    def _get_model_structure_from_current_position(self, current_position, id_record=None):
        """
        按照“先处理current_position及其所有后代，然后再向上逐级处理其父节点及其他子节点”的顺序，
        构建局部模型结构，并返回为格式化XML字符串。
        """
        import xml.etree.ElementTree as ET

        if id_record is None:
            id_record = set()

        if current_position not in self.id_element_map:
            raise ValueError(f"当前定位的 xmi:id {current_position} 不在模型中。")

        def clone_meta(elem):
            elem_id = self.get_attr(elem, "xmi:id")
            elem_type = self.get_attr(elem, "xmi:type") or ""
            elem_name = self.get_attr(elem, "name") or ""
            return ET.Element("packagedElement", attrib={
                "xmi:id": elem_id,
                "xmi:type": elem_type,
                "name": elem_name
            })

        def clone_descendants(src_elem, dst_elem):
            """
            递归克隆 src_elem 的后代（以 <packagedElement> 为主）
            """
            count = 0
            for child in src_elem:
                if self.get_attr(child, "xmi:type") == "uml:PackagedElement":
                    child_id = self.get_attr(child, "xmi:id")
                    if child_id and child_id not in id_record:
                        id_record.add(child_id)
                        new_child = clone_meta(child)
                        dst_elem.append(new_child)
                        count += 1
                        if len(id_record) < self.max_node_num:
                            count += clone_descendants(child, new_child)
                if len(id_record) >= self.max_node_num:
                    break
            return count

        # 构建最终返回的 XML 结构根节点
        root_summary = ET.Element("PartialModelStructure")

        # 1. 克隆 current_position 节点及其子树
        focus_elem = self.id_element_map[current_position]
        focus_id = self.get_attr(focus_elem, "xmi:id")
        id_record.add(focus_id)
        focus_node = clone_meta(focus_elem)
        root_summary.append(focus_node)
        clone_descendants(focus_elem, focus_node)

        # 2. 向上回溯父节点，依次克隆其其他子元素（排除已记录）
        current_elem = focus_elem
        while len(id_record) < self.max_node_num:
            # 找到当前元素的父节点
            parent_elem = None
            for candidate in self.project_xmi.iter():
                if current_elem in list(candidate):
                    parent_elem = candidate
                    break
            if parent_elem is None:
                break  # 已到达根节点

            parent_id = self.get_attr(parent_elem, "xmi:id")
            if not parent_id:
                break

            if parent_id not in id_record:
                id_record.add(parent_id)
                parent_clone = clone_meta(parent_elem)
                root_summary.append(parent_clone)
            else:
                # 已存在于 root_summary 中，找到已有的节点
                parent_clone = None
                for elem in root_summary:
                    if self.get_attr(elem, "xmi:id") == parent_id:
                        parent_clone = elem
                        break
                if parent_clone is None:
                    break

            # 在 parent_clone 下添加其其他子元素（除了 current_elem）
            for sibling in parent_elem:
                if sibling is current_elem:
                    continue
                if self.get_attr(sibling, "xmi:type") == "uml:PackagedElement":
                    sibling_id = self.get_attr(sibling, "xmi:id")
                    if sibling_id and sibling_id not in id_record:
                        id_record.add(sibling_id)
                        sibling_clone = clone_meta(sibling)
                        parent_clone.append(sibling_clone)
                        clone_descendants(sibling, sibling_clone)

                if len(id_record) >= self.max_node_num:
                    break

            current_elem = parent_elem

        return self.to_string(root_summary, pretty=True, xml_declaration=False)

        

    def _add_incremental_element(self,incremental_element,old_element):
        """
        incremental_element: 新增的模型元素
        old_element: 已存在的模型元素
        二者的id一样，本方法增量更新incremental_element的属性和子元素到old_element中
        如果新增的内容，old_element中不存在，则直接添加到old_element中；如果存在，则递归地调用本方法，处理同样的属性和子元素
        """
        # 1. 合并属性
        for key, value in incremental_element.attrib.items():
            if key not in old_element.attrib:
                old_element.set(key, value)
            # 若已存在，保留 old_element 中已有的值（可选策略）

        # 2. 为 old_element 的子元素构建 ID → element 映射
        old_children_id_map = {}
        for child in old_element:
            child_id = self.get_attr(child, 'xmi:id')
            if child_id:
                old_children_id_map[child_id] = child

        # 3. 遍历 incremental_element 的子元素，递归合并
        for inc_child in incremental_element:
            inc_child_id = self.get_attr(inc_child, 'xmi:id')
            if inc_child_id and inc_child_id in old_children_id_map:
                # 已存在，递归更新
                self._add_incremental_element(inc_child, old_children_id_map[inc_child_id])
            else:
                # 不存在，直接添加
                old_element.append(inc_child)
                self.id_element_map[inc_child_id] = inc_child  # 更新映射


    def add_incremental_elements(self, model_element_list, current_position=None):
        """
        将增量的模型元素添加到MagicDraw项目的XMI project_xmi 根元素中
        current_position: 可选参数，表示用户当前打开的模型的工作层级位置。用来让llm进行辅助
        """

        for incremental_elem in model_element_list:
            # 这里假设incremental_elem是一个Element对象
            # 将其添加到project_xmi的根元素下
            id_incremental = self.get_attr(incremental_elem, 'xmi:id')
            if id_incremental not in self.id_element_map:
                if not self.use_llm_structure: 
                    # 如果ID不存在于id_element_map中，直接添加到project_xmi
                    uml_Model = self.project_xmi.find('.//uml:Model', namespaces=self.namespace)
                    if uml_Model is None:
                        raise ValueError("未找到 uml:Model 元素，请检查XMI文件是否正确。")
                    uml_Model.append(incremental_elem)
                    # 更新id_element_map
                    self.id_element_map[id_incremental] = incremental_elem
                else:
                    # 如果use_llm_structure为True，则需要使用LLM来决定如何添加增量的元素
                    # 首先先确定当前以文档根节点的目录结构，能不能层级在max_node_num之内，如果能，则直接让llm决定插入到哪个id中。如果不能，则以current_position为中心元素，进行搜索。
                    num_packagedelment = len(self.project_xmi.findall('.//packagedElement'))
                    if num_packagedelment < self.max_node_num:
                        model_structure = self._get_model_structure()

                    else:
                        model_structure = self._get_model_structure_from_current_position(current_position)

                    incremental_elem_str = self.to_string(incremental_elem, pretty=True, xml_declaration=False)
                    target_id_response= self.llm.chat(self.prompt.format(
                        model_structure=model_structure,
                        current_position=current_position if current_position else "",
                        incremental_elem=incremental_elem_str
                    ))
                    if target_id_response:
                        import re
                        match = re.search(r"_\w{5,}", target_id_response)

                        if match:
                            print("Matched ID:", match.group(0))
                        else:
                            print("No valid ID found.")
                        id_target = match.group(0)
                        if id_target in self.id_element_map:
                            # 如果id_target存在，则将incremental_elem添加到id_target对应的元素下
                            target_element = self.id_element_map[id_target]
                            self._add_incremental_element(incremental_elem, target_element)
                        else:
                            # 如果id_target不存在，则直接添加到project_xmi的根元素下
                            self.project_xmi.append(incremental_elem)
                            # 更新id_element_map
                            self.id_element_map[id_incremental] = incremental_elem
                    else:
                        raise ValueError("LLM未返回有效的目标ID，请检查LLM配置或输入内容。")

            else:
                #如果id存在，则就要开始增量的更新了
                old_element = self.id_element_map[id_incremental]
                self._add_incremental_element(incremental_elem, old_element)




    def wrap_project(self, new_content,output_file=None,current_position=None,save=True):
        """
        将新的内容包装到MagicDraw项目中
        """
        if self.content_type == ContentType.JSON:
            pass 
        elif self.content_type == ContentType.XMI:
            success, model_list, stereo_list = self.get_model_and_stereotypes(new_content) 
            if not success:
                raise ValueError("XMI内容验证失败，无法包装项目。原因是 XML语法未通过验证。")
            
            for stereo in stereo_list:
                self.project_xmi.append(stereo)
            
            self.add_incremental_elements(model_list, current_position)
        self.project_data =self.to_string(self.project_xmi,pretty=True)  
        if save:
            output_file = output_file or self.project_path.replace('.xml', "_wrapped.xml")

            self.save_project(output_file) 
        else:
            print("Project data not saved, but ready for further processing.")
            return self.project_data



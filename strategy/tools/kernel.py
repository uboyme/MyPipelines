# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
import os

import pandas as pd
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter

def retrive_sysml_standard(query: str) -> str:
    """
    Retrieve the SysML standard content based on the query.
    :param query: The query string to search for the SysML standard content.
    :return: The SysML standard content as a string.
    """ 
    print("Retrieving SysML standard content for query:", query)

    persist_path = "dataset/rag_content/index/sysml_standard"

    # 若 index 存在，加载；否则构建并保存
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_path)
        index = load_index_from_storage(storage_context)
    except:
        docs = SimpleDirectoryReader("./dataset/rag_content/standard").load_data()
        # docs = SimpleDirectoryReader("OMGSysML.pdf").load_data()
        # index = VectorStoreIndex.from_documents(docs)
        splitter = SentenceSplitter(chunk_size=8192, chunk_overlap=0)
        index = VectorStoreIndex.from_documents(
            docs,
            transformations=[splitter]
        )
        index.storage_context.persist(persist_dir=persist_path)

    retriever = index.as_retriever(similarity_top_k=3)  # 设置检索数量

    nodes = retriever.retrieve(query)
    context_str = "\n\n".join([n.node.get_content() for n in nodes])

    return context_str

from pathlib import Path

def retrive_sysml_rules(query: str) -> str:
    """
    Retrieve SysML model generation rules based on the query. When the query is related to SysML modeling, it returns relevant rules and the Agent should call this tool to retrieve the rules.
    For example, rules include ACT-Activity,ACT-TestCase,BDD-Block,BDD-ConstraintBlock,BDD-InterfaceBlock,BDD-ValueType, IBD-BindingConnector, IBD-Connector
    IBD-InformationFlow, IBD-ItemFlow, PAR-caused_by, PKG-Package, REQ-DeriveReqt, REQ-Requirement, REQ-Trace, SEQ-Interaction, STM-StateMachine, UCD-Stakeholder, UCD-UseCase.
    :param query: The query string to search for SysML rules.
    :return: The SysML generation rules as a string.
    """
    print("Retrieving SysML rules for query:", query)

    persist_path = "dataset/rag_content/index/sysml_rules"

    # 若 index 存在，加载；否则构建并保存
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_path)
        index = load_index_from_storage(storage_context)
    except:
        rule_dir=Path("./dataset/rag_content/sysml-rules")
        rule_texts=[]
        for file in rule_dir.glob("*.txt"):
            with open(file, "r", encoding="utf-8") as f:
                content=f.read().strip()
                rule_texts.append({"text": content, "source": file.name})
        from llama_index.core import Document
        docs = [Document(text=rule["text"], metadata={"source": rule["source"]}) for rule in rule_texts]
        # docs = SimpleDirectoryReader("./dataset/rag_content/rules").load_data()
        splitter = SentenceSplitter(chunk_size=8192, chunk_overlap=0)
        index = VectorStoreIndex.from_documents(
            docs,
            transformations=[splitter]
        )
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir=persist_path)


    retriever = index.as_retriever(similarity_top_k=3)  # 设置检索数量

    nodes = retriever.retrieve(query)
    rule_str = "\n\n".join([n.node.get_content() for n in nodes])
    print("Retrieved SysML rules:", rule_str)
    return rule_str








def retrive_few_shot_examples(query: str) -> str:
    """
    if query is related to SysML modeling or generating, it must call this tool to retrieve few-shot examples to learn how to represent a sysml model.
    Retrieve few-shot examples based on the query. When the query is related to SysML modeling, it returns relevant few-shot examples and the Agent should call this tool to retrieve the examples.
    
    if query is not related to SysML modeling, the Agent should not call this tool.

    :param query: The query string to search for few-shot examples.
    :return: The few-shot examples（sysml model fragement） as a string. it contains requirement, related_element(context) and focus_element(target model), 
    """
    print("Retrieving few-shot examples for query:", query)

    def load_fewshot_docs_from_dir(base_dir):
        fewshot_docs = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".csv"):
                    #file的所在最近的父目录
                    # dir_name = os.path.basename(file)
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path)
                    for i, row in df.iterrows():
                        text = f"requirement: {row['requirement']}\n related_elements: {row['related_elements']}\n focus_element: {row['focus_element']}"
                        # print("Loading Few-shot Document:", text)
                        # print("file:", file[:-16], "id:", row["id"]    )
                        fewshot_docs.append(Document(text=text, metadata={"id": row["id"], "Element-Type": file[:-16]}))
        return fewshot_docs
    persist_path = "dataset/rag_content/index/few_shot_examples"
    # 若 index 存在，加载；否则构建并保存
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_path)
        fewshot_index = load_index_from_storage(storage_context)
    except:
        # docs = SimpleDirectoryReader("./dataset/rag_content/few_shot_examples").load_data()
        # index = VectorStoreIndex.from_documents(docs)
            # 使用
        base_dir = "./dataset/dataset_native-xmi-csv/test_set"  # 你的few-shot csv根目录
        fewshot_docs = load_fewshot_docs_from_dir(base_dir)
        splitter = SentenceSplitter(chunk_size=8192, chunk_overlap=0)
        fewshot_index = VectorStoreIndex.from_documents(
            fewshot_docs,
            transformations=[splitter]
        )
        # fewshot_index = VectorStoreIndex.from_documents(fewshot_docs)
        fewshot_index.storage_context.persist(persist_dir=persist_path)


    retriver = fewshot_index.as_retriever(similarity_top_k=2)  # 设置检索数量

    nodes = retriver.retrieve(query)
    # 判断是否检索为空

    
    content_str = "\n\n".join([n.node.get_content() for n in nodes])

    if not content_str.strip():
        print("[Fallback] 未匹配到相关 few-shot 示例，返回默认样例。")
        content_str = """
requirement:
The spacecraft shall receive and execute ground commands, collect and return observation data through its sensors, provide telemetry, maintain operational state via attitude control, thermal management, power provision, and fault handling, and interface with external systems including radiation sources, propulsion, and ground communications.
==========================================
related_elements:
<packagedElement xmi:id="_18_0_1_ae502ce_1419959276703_281931_46369" name="Shape" xmi:type="uml:DataType"/>
<packagedElement xmi:id="_18_0_1_ae502ce_1413163385985_440939_45975" name="GndCmd&amp;Data I/F" xmi:type="uml:Class"/>
<packagedElement xmi:id="_18_0_1_ae502ce_1409014388334_928579_19192" name="LV Elecrictal I/F" xmi:type="uml:Class"/>
<packagedElement xmi:id="_18_0_1_ae502ce_1408796371990_968654_21768" name="Orbit" xmi:type="uml:Class"/>
<packagedElement xmi:id="_18_0_1_ae502ce_1408728025445_333658_20841" name="Electrical Energy" xmi:type="uml:DataType"/>
<packagedElement xmi:id="_18_0_1_ae502ce_1416137099118_962721_45677" name="Solar Array Deployment Initiate" xmi:type="uml:Signal"/>


=========================================
focus_element:
<packagedElement xmi:type="uml:Class" xmi:id="_17_0_5_ae502ce_1403989559103_73588_11766" name="Spacecraft" classifierBehavior="_17_0_5_ae502ce_1404309694118_920434_11858">
  <ownedBehavior xmi:type="uml:StateMachine" xmi:id="_17_0_5_ae502ce_1404309694118_920434_11858" name="Spacecraft States">
                        </ownedBehavior>
  <ownedAttribute xmi:type="uml:Property" xmi:id="_17_0_5_ae502ce_1403990423302_193949_12344" name="mass" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Property" xmi:id="_17_0_5_ae502ce_1404255086980_720308_13324" name="size" visibility="public" aggregation="composite" type="_18_0_1_ae502ce_1419959276703_281931_46369"/>
  <ownedAttribute xmi:type="uml:Property" xmi:id="_17_0_5_ae502ce_1404255078806_195359_13319" name="power" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Property" xmi:id="_18_0_1_ae502ce_1408727824142_927105_20835" name="deltaV" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Property" xmi:id="_18_0_1_ae502ce_1409409335658_648119_13597" name="max radiation level" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Property" xmi:id="_18_0_1_ae502ce_1414337275002_27455_45071" name="reliability" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Property" xmi:id="_17_0_5_ae502ce_1404255114577_989452_13329" name="life" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_17_0_5_ae502ce_1404255524468_713976_13350" name="solar radiation i/f" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_17_0_5_ae502ce_1404255547790_98000_13382" name="em radiation i/f" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_17_0_5_ae502ce_1404255588319_863050_13464" name="observation sensor i/f" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_17_0_5_ae502ce_1404255611360_753302_13514" name="thrust i/f" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_17_0_5_ae502ce_1404255825258_589871_13694" name="gnd cmd &amp; data i/f" visibility="public" aggregation="composite" type="_18_0_1_ae502ce_1413163385985_440939_45975"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_17_0_5_ae502ce_1404255867425_397747_13766" name="LV electrical i/f" visibility="public" aggregation="composite" type="_18_0_1_ae502ce_1409014388334_928579_19192"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_17_0_5_ae502ce_1404255880014_234106_13842" name="LV mechanical i/f" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Property" xmi:id="_17_0_5_ae502ce_1404255137493_319947_13334" name="cost" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_18_0_1_ae502ce_1408725488145_939550_20504" name="thermal radiation i/f" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Property" xmi:id="_17_0_5_ae502ce_1404255186773_859983_13339" name="data capacity" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Property" xmi:id="_18_0_1_ae502ce_1408796821756_623506_21930" name="orbit" visibility="public" aggregation="shared" type="_18_0_1_ae502ce_1408796371990_968654_21768"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_18_0_1_ae502ce_1408929252190_147980_13509" name="star tracker i/f" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_18_0_1_ae502ce_1408929858595_644998_13677" name="inertial sensor i/f" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_18_0_1_ae502ce_1409015015609_166750_19490" name="impact i/f" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_18_0_1_ae502ce_1409107015518_254722_15379" name="gps i/f" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_18_0_1_ae502ce_1409108007126_945424_15774" name="horizon tracker i/f" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_18_0_1_ae502ce_1409318805475_688895_14338" name="drag i/f" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Property" xmi:id="_17_0_5_ae502ce_1404255222046_224114_13344" name="probability of detection" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Property" xmi:id="_18_0_1_ae502ce_1420059171255_35157_48165" name="probability of false alarm" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Property" xmi:id="_18_0_1_ae502ce_1416092602859_524190_50103" name="pointing accuracy" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Property" xmi:id="_18_0_1_ae502ce_1408728588827_793750_20997" name="e" visibility="public" aggregation="shared" type="_18_0_1_ae502ce_1408728025445_333658_20841"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_18_0_1_ae502ce_1421696771683_317256_47417" name="sun tracker i/f" visibility="public" aggregation="composite"/>
  <ownedAttribute xmi:type="uml:Port" xmi:id="_18_0_1_ae502ce_1421696781979_210108_47418" name="magnetometer i/f" visibility="public" aggregation="composite"/>
  <ownedOperation xmi:type="uml:Operation" xmi:id="_17_0_5_ae502ce_1403990232032_419620_12338" name="collect observation data" visibility="public"/>
  <ownedOperation xmi:type="uml:Operation" xmi:id="_17_0_5_ae502ce_1404253889302_586438_13191" name="return observation data" visibility="public"/>
  <ownedOperation xmi:type="uml:Operation" xmi:id="_18_0_1_ae502ce_1413647982680_712444_44688" name="receive ground command" visibility="public"/>
  <ownedOperation xmi:type="uml:Operation" xmi:id="_17_0_5_ae502ce_1403990192970_583359_12334" name="provide telemetry data"/>
  <ownedOperation xmi:type="uml:Operation" xmi:id="_17_0_5_ae502ce_1403990159252_53712_12330" name="control attitude" visibility="public"/>
  <ownedOperation xmi:type="uml:Operation" xmi:id="_17_0_5_ae502ce_1404254518921_513644_13233" name="control acceleration)"/>
  <ownedOperation xmi:type="uml:Operation" xmi:id="_17_0_5_ae502ce_1403990170782_642694_12332" name="control thermal environment" visibility="public"/>
  <ownedOperation xmi:type="uml:Operation" xmi:id="_17_0_5_ae502ce_1403990112792_324990_12326" name="provide electrical power" visibility="public"/>
  <ownedOperation xmi:type="uml:Operation" xmi:id="_17_0_5_ae502ce_1403990082554_629031_12324" name="manage faults" visibility="public"/>
  <ownedOperation xmi:type="uml:Operation" xmi:id="_17_0_5_ae502ce_1404253582743_415549_13165" name="control separation" visibility="public"/>
  <ownedOperation xmi:type="uml:Operation" xmi:id="_17_0_5_ae502ce_1403990264742_653394_12340" name="provide structural integrity" visibility="public"/>
  <ownedOperation xmi:type="uml:Operation" xmi:id="_18_0_1_ae502ce_1419012952526_619684_45233" name="deploy antenna" visibility="public"/>
  <ownedOperation xmi:type="uml:Operation" xmi:id="_18_0_1_ae502ce_1419012964164_595980_45234" name="deploy solar array" visibility="public"/>
  <ownedReception xmi:type="uml:Reception" xmi:id="_18_1_ae502ce_1424290149410_42761_48086" name="Solar Array Deployment Command" visibility="public" signal="_18_0_1_ae502ce_1416137099118_962721_45677"/>
</packagedElement>
<sysml:Block xmi:id="_17_0_5_ae502ce_1403989559103_405186_11767" base_Class="_17_0_5_ae502ce_1403989559103_73588_11766"/>
<MD_Customization_for_SysML__additional_stereotypes:ValueProperty xmi:id="_18_0_1_ae502ce_1409409335674_233116_13598" base_Property="_18_0_1_ae502ce_1409409335658_648119_13597"/>
<MD_Customization_for_SysML__additional_stereotypes:ValueProperty xmi:id="_17_0_5_ae502ce_1404255114577_990530_13330" base_Property="_17_0_5_ae502ce_1404255114577_989452_13329"/>
<MD_Customization_for_SysML__additional_stereotypes:ReferenceProperty xmi:id="_18_0_1_ae502ce_1408796821787_536390_21938" base_Property="_18_0_1_ae502ce_1408796821756_623506_21930"/>
<MD_Customization_for_SysML__additional_stereotypes:ValueProperty xmi:id="_18_0_1_ae502ce_1408727824142_780535_20836" base_Property="_18_0_1_ae502ce_1408727824142_927105_20835"/>
<MD_Customization_for_SysML__additional_stereotypes:ValueProperty xmi:id="_18_0_1_ae502ce_1414337275002_37724_45072" base_Property="_18_0_1_ae502ce_1414337275002_27455_45071"/>
<MD_Customization_for_SysML__additional_stereotypes:ValueProperty xmi:id="_18_0_1_ae502ce_1420059171255_348223_48166" base_Property="_18_0_1_ae502ce_1420059171255_35157_48165"/>
<MD_Customization_for_SysML__additional_stereotypes:ValueProperty xmi:id="_18_0_1_ae502ce_1408728588889_47259_21005" base_Property="_18_0_1_ae502ce_1408728588827_793750_20997"/>
<MD_Customization_for_SysML__additional_stereotypes:ValueProperty xmi:id="_17_0_5_ae502ce_1404255078806_1745_13320" base_Property="_17_0_5_ae502ce_1404255078806_195359_13319"/>
<MD_Customization_for_SysML__additional_stereotypes:ValueProperty xmi:id="_17_0_5_ae502ce_1404255186773_534433_13340" base_Property="_17_0_5_ae502ce_1404255186773_859983_13339"/>
<MD_Customization_for_SysML__additional_stereotypes:ValueProperty xmi:id="_17_0_5_ae502ce_1404255086980_303054_13325" base_Property="_17_0_5_ae502ce_1404255086980_720308_13324"/>
<MD_Customization_for_SysML__additional_stereotypes:ValueProperty xmi:id="_17_0_5_ae502ce_1404255137493_979290_13335" base_Property="_17_0_5_ae502ce_1404255137493_319947_13334"/>
<MD_Customization_for_SysML__additional_stereotypes:ValueProperty xmi:id="_17_0_5_ae502ce_1404255222046_727868_13345" base_Property="_17_0_5_ae502ce_1404255222046_224114_13344"/>
<MD_Customization_for_SysML__additional_stereotypes:ValueProperty xmi:id="_17_0_5_ae502ce_1403990423302_989379_12345" base_Property="_17_0_5_ae502ce_1403990423302_193949_12344"/>
<MD_Customization_for_SysML__additional_stereotypes:ValueProperty xmi:id="_18_0_1_ae502ce_1416092602861_204484_50104" base_Property="_18_0_1_ae502ce_1416092602859_524190_50103"/>

"""

    content_str =""" In the XMI representation of SysML, SysML elements must be explicitly encapsulated based on UML, meaning that an additional semantic layer (such as stereotypes) is required to represent these SysML-specific constructs. \n""" +content_str
    return content_str
import re
# 导入 MagicDrawProjectLoader
from project.project_loader import MagicDrawProjectLoader
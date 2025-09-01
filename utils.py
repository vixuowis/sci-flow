import os
from obs import ObsClient
import streamlit as st
from streamlit_tree_select import tree_select

def list_objects(prefix, client, bucket, delimiter="/"):
    """
    列出指定桶中指定前缀下的所有对象（文件），返回文件 key 的列表。

    Args:
        prefix (str): 对象前缀，即想要列出的“路径”
        client (ObsClient): 已初始化的 ObsClient 对象
        bucket (str): 桶名
        delimiter (str, optional): 路径分隔符，默认 "/"。设置 delimiter 可以模拟目录结构

    Returns:
        list[str]: 指定前缀下所有对象的 key 字符串列表
    """
    all_files = []  # 保存所有对象 key
    marker = ""     # 分页标记

    while True:
        # 调用 listObjects API 获取当前批次对象
        resp = client.listObjects(
            bucketName=bucket,
            prefix=prefix,
            delimiter=delimiter,
            marker=marker
        )

        # 判断响应状态
        if resp.status >= 300:
            print(f"错误: {resp.errorCode}, {resp.errorMessage}")
            break

        # 获取当前批次的对象列表
        contents = getattr(resp.body, "contents", [])
        if contents:
            for obj in contents:
                all_files.append(obj.key)  # 保存文件 key

        # 分页处理
        if getattr(resp.body, "isTruncated", False):
            marker = getattr(resp.body, "nextMarker", "")
        else:
            break

    return all_files




# ================= OBS 获取文件函数 =================
def list_all_objects(prefix, client, bucket, max_depth=None):
    def count_depth(key):
        return len(key.strip("/").split("/"))

    items = []
    marker = ""

    while True:
        resp = client.listObjects(
            bucketName=bucket,
            prefix=prefix,
            marker=marker
        )
        if resp.status >= 300:
            st.error(f"OBS 错误: {resp.errorCode}, {resp.errorMessage}")
            break

        for obj in getattr(resp.body, "contents", []) or []:
            if max_depth is not None and count_depth(obj.key) > count_depth(prefix) + max_depth:
                continue
            items.append(obj.key)

        if getattr(resp.body, "isTruncated", False):
            marker = getattr(resp.body, "nextMarker", "")
        else:
            break

    return items

# ================= 将对象列表转成树结构 =================
def build_tree(keys, prefix=""):
    """
    将 OBS 对象 key 列表转换为 streamlit_tree_select 所需的树形节点
    value 使用完整路径保证唯一性
    """
    tree = {}

    for key in keys:
        parts = key[len(prefix):].strip("/").split("/")
        current = tree
        for i, part in enumerate(parts):
            if part not in current:
                # value 使用完整路径
                node_value = "/".join(parts[:i+1])
                current[part] = {"__value": node_value}
            current = current[part]

    def dict_to_nodes(d):
        nodes = []
        for k, v in d.items():
            node = {"label": k, "value": v["__value"]}
            # 排除 __value 避免被当成子节点
            children_dict = {kk: vv for kk, vv in v.items() if kk != "__value"}
            if children_dict:
                node["children"] = dict_to_nodes(children_dict)
            nodes.append(node)
        return nodes

    return dict_to_nodes(tree)

if __name__ == "__main__":
    # 配置
    ENDPOINT = "https://obs.cn-north-4.myhuaweicloud.com"  # 华北-北京四
    BUCKET = "gaoyuan-49d0"
    PREFIX = "石冰川数据-遥感+无人机/鲁朗石冰川_1107/001/MSS/"
    DELIMITER = "/"  # 模拟目录

    # 初始化客户端（如果桶是公共读，AK/SK 可不填）
    CLIENT = ObsClient(server=ENDPOINT)

    # 调用函数，获取文件列表
    files = list_objects(PREFIX, CLIENT, BUCKET, DELIMITER)

    # 输出文件数量和部分示例
    print(f"共找到 {len(files)} 个文件")
    for f in files[:10]:  # 仅显示前 10 个示例
        print(f)

    # 关闭客户端
    CLIENT.close()
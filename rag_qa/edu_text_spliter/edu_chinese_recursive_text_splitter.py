# -*- coding: utf-8 -*-
"""
中文递归字符文本切分器（ChineseRecursiveTextSplitter）
======================================================
本文件服务于 RAG 问答系统的「文档切块」环节：
把一篇超长的中文文档，按语义边界（段落 -> 句子 -> 分句）递归切成多个小块，
便于后续做 embedding 向量化与检索。

整体设计思路
------------
1. 继承 LangChain 的 RecursiveCharacterTextSplitter，复用父类已经实现好的
   chunk_size（块大小）、chunk_overlap（重叠长度）、_merge_splits（合并+重叠）等机制。
2. 针对中文语法，重写了分隔符优先级列表，并默认开启两个与父类不同的开关：
     keep_separator=True     切分时保留标点，让每个块都是完整句子；
     is_separator_regex=True 分隔符按正则表达式解析（而不是字面量）。
3. 切分策略：从优先级最高的分隔符开始尝试，命中后切分；对仍超长的块，
   递归使用更低优先级（更细粒度）的分隔符继续切，直到所有块都小于
   chunk_size，或分隔符已全部用尽。
"""

import re  # 正则表达式库：用于按分隔符切分文本、压缩连续换行
from typing import List, Optional, Any  # 类型标注：List=列表、Optional=可空、Any=任意类型
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 父类：LangChain 递归字符切分器
import logging  # 日志库：便于调试时输出切分过程的信息

# 创建本模块的日志记录器，之后可用 logger.debug(...) / logger.warning(...) 等向日志系统输出
logger = logging.getLogger(__name__)


def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    """
    用正则分隔符把一段文本切成列表。

    与父类 langchain 内部的 _split_text_with_regex 逻辑基本一致。
    函数名中的 from_end 表示：当需要保留分隔符时，分隔符会被拼在
    每一段文本的【末尾】（而不是开头），保证标点归属在前一句话上。

    参数:
        text:           待切分的原始文本
        separator:      分隔符（正则表达式字符串）
        keep_separator: 是否把分隔符保留在切分结果中
    返回:
        切分后的字符串列表（已过滤掉空串）
    """
    # —— 分支 1：分隔符非空，走正则切分 ——
    if separator:
        # —— 子分支 1.1：需要保留分隔符（标点）——
        if keep_separator:
            # re.split 中给模式加括号 "(...)" 会把分隔符也保留进结果数组。
            # 例：re.split("(。)", "你好。再见。") -> ["你好", "。", "再见", "。", ""]
            _splits = re.split(f"({separator})", text)
            # 此时 _splits 元素交替出现：[文本0, 分隔符0, 文本1, 分隔符1, ..., 文本N]。
            #   _splits[0::2] 取偶数下标 => 所有文本段
            #   _splits[1::2] 取奇数下标 => 所有分隔符段
            # zip 把两者一一配对，再用 "".join 拼起来，
            # 得到 ["文本0分隔符0", "文本1分隔符1", ...]，
            # 即每段文本后面紧跟着它自己的分隔符（标点被保留在句末）。
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            # 若 _splits 长度是奇数，说明原文本以「文本」结尾、而不是以「分隔符」结尾，
            # 最后那一段文本没有配对的分隔符，需要单独补进结果列表。
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
        # —— 子分支 1.2：不保留分隔符 ——
        else:
            # 直接按分隔符切掉，分隔符（标点）被丢弃，不进入结果
            splits = re.split(separator, text)
    # —— 分支 2：分隔符为空，退化为逐字符切分 ——
    else:
        # 把文本拆成单字符列表（list("你好") -> ["你", "好"]）
        splits = list(text)
    # 过滤掉空字符串（例如相邻分隔符、或文本首尾切出来的空串）
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """
    中文递归文本切分器。

    相比父类 RecursiveCharacterTextSplitter，主要差异：
      1. 内置了适配中文语法的分隔符优先级列表；
      2. 默认 keep_separator=True（保留标点）和 is_separator_regex=True（按正则切）。
    """

    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """
        构造一个切分器。

        参数:
            separators:         自定义分隔符列表（按优先级从高到低排列）。
                                不传则使用本类内置的中文分隔符列表。
            keep_separator:     切分时是否保留分隔符（标点）。默认 True。
            is_separator_regex: 分隔符是否按正则表达式解析。默认 True。
            **kwargs:           其余参数透传给父类，常见的有：
                                chunk_size   = 每个块的目标大小（字符数）
                                chunk_overlap = 相邻块之间的重叠长度（字符数）
        """
        # 调用父类构造函数，把 keep_separator 一并传进去（父类内部会用它控制切分行为）。
        super().__init__(keep_separator=keep_separator, **kwargs)
        # 设置分隔符列表：调用方未传时，使用下面这份「中文专用」默认列表。
        # 列表顺序即优先级：越靠前越先尝试、粒度越粗。
        self._separators = separators or [
            "\n\n",           # 1. 连续两个换行 -> 段落边界（优先级最高，先按段落切）
            "\n",             # 2. 单个换行 -> 行边界
            "。|！|？",        # 3. 中文句末标点：句号 / 叹号 / 问号
            "\.\s|\!\s|\?\s", # 4. 英文句末标点 + 空格（\s 要求后面跟空白，避免误切缩写与小数点的点号）
            "；|;\s",         # 5. 中文分号，或英文分号 + 空格
            "，|,\s"          # 6. 中文逗号，或英文逗号 + 空格（优先级最低，最后尝试）
        ]
        # 记录分隔符是否为正则表达式，供 _split_text 判断是否需要对分隔符做 re.escape 转义
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        核心切分方法：把一段文本切成若干块，并返回结果列表。

        算法流程（递归 + 合并）：
          1. 从 separators 中按优先级找到第一个在文本里真实出现的分隔符；
          2. 用该分隔符把文本切分；
          3. 对每一段：
               长度 < chunk_size  -> 攒进待合并缓冲区；
               长度 >= chunk_size -> 先合并落盘缓冲区，再对该段递归
                                     （递归时只传更低优先级的分隔符）；
          4. 收尾：合并缓冲区剩余块，统一清理（去空、去首尾空白、压缩连续换行）。
        """
        final_chunks = []  # 最终结果：保存所有已确定、可返回的切块
        # 先用最后一个分隔符作为兜底值（若文本里一个分隔符都匹配不到，就用它来切）
        separator = separators[-1]
        # 记录当前层选中分隔符之后的「更低优先级」分隔符，递归时传给下一层
        new_separators = []
        # —— Step 1：按优先级探测，选出第一个真实出现于文本的分隔符 ——
        for i, _s in enumerate(separators):
            # 若非正则模式，需先把分隔符 re.escape 转义成「字面量匹配」的正则表达式
            _separator = _s if self._is_separator_regex else re.escape(_s)
            # 空字符串分隔符表示「到此为止」：直接采用空分隔符（逐字符切分），并停止探测
            if _s == "":
                separator = _s
                break
            # 若该分隔符在文本中命中了，选中它，并把列表中排在它后面的
            # 所有分隔符都保留为下一层的候选（它们优先级更低、粒度更细）。
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break
        # 转成本次实际使用的正则表达式
        _separator = separator if self._is_separator_regex else re.escape(separator)
        # —— Step 2：用选中的分隔符切分文本（keep_separator 决定标点去留）——
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # —— Step 3：合并短块 + 递归切长块 ——
        _good_splits = []  # 缓冲区：攒住所有「足够短、无需再切」的块
        # 合并短块时，块与块之间用什么字符串连接：
        #   keep_separator=True 时，标点已经保留在每块末尾，用空串直接拼接即可，
        #   避免把标点重复一遍；
        #   否则用原始分隔符拼接，把被切掉的标点语义补回来。
        _separator = "" if self._keep_separator else separator
        # 遍历切分出的每一段
        for s in splits:
            # —— 情况 A：这段长度 < chunk_size，属于「短块」，攒进缓冲区 ——
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            # —— 情况 B：这段仍然太长，需要进一步处理 ——
            else:
                # 先把缓冲区里攒下的短块合并、落盘
                #（_merge_splits 是父类方法，负责把短块拼到 chunk_size 上限，并处理 chunk_overlap 重叠）
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []  # 缓冲区已清空，重新开始累积
                # 若更低优先级的分隔符已经用尽，无法再细分 -> 只能整段保留
                if not new_separators:
                    final_chunks.append(s)
                # 否则递归调用自身，用更细粒度的分隔符继续切这段长文本
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        # —— Step 4：收尾 ——
        # 循环结束后，缓冲区里可能还攒着最后一批短块，也要合并、落盘
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        # 统一清理每个块：
        #   chunk.strip()             去掉块首尾的空白字符
        #   re.sub(r"\n{2,}", "\n", ...) 把连续多个换行压缩成单个换行，让块更紧凑
        #   if chunk.strip() != ""    过滤掉切分过程中产生的空块
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip() != ""]


if __name__ == "__main__":
    # —— 自测代码：直接运行本文件时（python edu_chinese_recursive_text_splitter.py）执行 ——
    # 构造一个中文切分器实例，用于下面的人工验证
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,      # 切分时保留标点
        is_separator_regex=True,  # 分隔符按正则表达式解析
        chunk_size=150,           # 每个块目标大小约为 150 字
        chunk_overlap=10          # 相邻块之间重叠 10 字，避免切分点处丢失上下文语义
    )
    # 测试用的原始文本：一份中国对外贸易形势报告的开头段落（多段、含中英文标点与换行）
    ls = [
        """中国对外贸易形势报告（75页）。前 10 个月，一般贸易进出口 19.5 万亿元，增长 25.1%， 比整体进出口增速高出 2.9 个百分点，占进出口总额的 61.7%，较去年同期提升 1.6 个百分点。其中，一般贸易出口 10.6 万亿元，增长 25.3%，占出口总额的 60.9%，提升 1.5 个百分点；进口8.9万亿元，增长24.9%，占进口总额的62.7%， 提升 1.8 个百分点。加工贸易进出口 6.8 万亿元，增长 11.8%， 占进出口总额的 21.5%，减少 2.0 个百分点。其中，出口增 长 10.4%，占出口总额的 24.3%，减少 2.6 个百分点；进口增 长 14.2%，占进口总额的 18.0%，减少 1.2 个百分点。此外， 以保税物流方式进出口 3.96 万亿元，增长 27.9%。其中，出 口 1.47 万亿元，增长 38.9%；进口 2.49 万亿元，增长 22.2%。前三季度，中国服务贸易继续保持快速增长态势。服务 进出口总额 37834.3 亿元，增长 11.6%；其中服务出口 17820.9 亿元，增长 27.3%；进口 20013.4 亿元，增长 0.5%，进口增 速实现了疫情以来的首次转正。服务出口增幅大于进口 26.8 个百分点，带动服务贸易逆差下降 62.9%至 2192.5 亿元。服 务贸易结构持续优化，知识密集型服务进出口 16917.7 亿元， 增长 13.3%，占服务进出口总额的比重达到 44.7%，提升 0.7 个百分点。 二、中国对外贸易发展环境分析和展望 全球疫情起伏反复，经济复苏分化加剧，大宗商品价格 上涨、能源紧缺、运力紧张及发达经济体政策调整外溢等风 险交织叠加。同时也要看到，我国经济长期向好的趋势没有 改变，外贸企业韧性和活力不断增强，新业态新模式加快发 展，创新转型步伐提速。产业链供应链面临挑战。美欧等加快出台制造业回迁计 划，加速产业链供应链本土布局，跨国公司调整产业链供应 链，全球双链面临新一轮重构，区域化、近岸化、本土化、 短链化趋势凸显。疫苗供应不足，制造业“缺芯”、物流受限、 运价高企，全球产业链供应链面临压力。 全球通胀持续高位运行。能源价格上涨加大主要经济体 的通胀压力，增加全球经济复苏的不确定性。世界银行今年 10 月发布《大宗商品市场展望》指出，能源价格在 2021 年 大涨逾 80%，并且仍将在 2022 年小幅上涨。IMF 指出，全 球通胀上行风险加剧，通胀前景存在巨大不确定性。""",
    ]
    # 逐条对测试文本执行切分，并打印结果
    for inum, text in enumerate(ls):
        print(inum)  # 打印当前文本的编号（索引），方便把输出对应到输入
        chunks = text_splitter.split_text(text)  # 调用父类对外接口执行切分，返回切块列表
        for chunk in chunks:
            print(chunk)      # 打印每一个切块的内容
            print('*' * 80)   # 打印 80 个 * 作为块与块之间的视觉分隔线

# # -*- coding: utf-8 -*-
# """
# yield 教学演示程序
# 从浅入深演示 yield 的核心概念：
# 1. 生成器函数 vs 普通函数
# 2. yield 的"暂停-恢复"执行机制
# 3. next() 手动驱动生成器
# 4. 生成器中的 return（StopIteration 的坑）
# 5. 惰性计算：用生成器处理"无限"数据
# 6. 实际应用：模拟 LLM 流式输出
# """
#
# def yield_test():
#     yield 1111
#     yield 2222
#     yield 3333
#     yield 4444
#
# yt = yield_test()
#
# for y in yt:
#     print(y)

#
import time


def print_section(title):
    """打印课程分隔线"""
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)
#
#
# # ============================================================
# # 第 1 课：生成器函数 vs 普通函数
# # ============================================================
# def normal_func():
#     """普通函数：一次性算完所有结果，装进列表返回"""
#     results = []
#     for i in range(3):
#         results.append(i * i)
#     return results
#
#
# def generator_func():
#     """生成器函数：每 yield 一次，产出一个结果"""
#     for i in range(3):
#         yield i * i
#
#
# print_section("第 1 课：生成器函数 vs 普通函数")
#
# r1 = normal_func()
# r2 = generator_func()
# print(f"普通函数返回：{r1}    <-- 直接得到一个列表")
# print(f"生成器函数返回：{r2}  <-- 只得到一个生成器对象！")
# print(f"生成器需要迭代才能取出值：{list(r2)}")
#
# #
# print("\n>>> 关键点：调用生成器函数时，函数体【一行都没有执行】，")
# print(">>> 只是创建了一个生成器对象，真正的计算发生在迭代时。")

#
# # ============================================================
# # 第 2 课：yield 的"暂停-恢复"机制
# # ============================================================
# def demo_pause_resume():
#     print("    [生成器] 开始执行")
#     print("    [生成器] 准备 yield 1 ...")
#     yield 1
#     print("    [生成器] 从上次暂停处恢复！准备 yield 2 ...")
#     yield 2
#     print("    [生成器] 又恢复了！准备 yield 3 ...")
#     yield 3
#     print("    [生成器] 没有更多 yield 了，函数结束")
#
#
# print_section("第 2 课：yield 的暂停-恢复机制（注意打印顺序）")
#
# gen = demo_pause_resume()
# print("[主程序] 生成器已创建，但注意上面没有任何 [生成器] 的打印！")
# print("[主程序] 开始迭代：")
# # for value in gen:
# #     print(f"[主程序] 收到了值：{value}")
# print(f"[主程序] 收到了值：{next(gen)}")
# print(f"[主程序] 收到了值：{next(gen)}")
#
# print("\n>>> 关键点：每遇到 yield，生成器【冻结】现场并交出值；")
# print(">>> 下次迭代时从冻结处【原样恢复】继续执行。")


# # ============================================================
# # 第 3 课：用 next() 手动驱动生成器
# # ============================================================
# print_section("第 3 课：next() 手动驱动 与 耗尽后的 StopIteration")
#
# gen = demo_pause_resume()
# print(f"第 1 次 next：{next(gen)}")
# print(f"第 2 次 next：{next(gen)}")
# print(f"第 3 次 next：{next(gen)}")
# try:
#     next(gen)  # 第 4 次：已经耗尽
# except StopIteration:
#     print("第 4 次 next：抛出 StopIteration 异常（生成器耗尽）")
#
# print("\n>>> 关键点：for 循环内部其实就是在反复调用 next()，")
# print(">>> 并自动捕获 StopIteration 来结束循环。")
#
#
# # ============================================================
# # 第 4 课：实战 —— 模拟 LLM 流式输出（呼应 DashScope 代码）
# # ============================================================
def mock_stream_answer(full_answer):
    """模拟大模型逐 token 输出：边生成边 yield"""
    collected_content = ""
    for char in full_answer:
        collected_content += char
        time.sleep(1.0)
        yield char                  # 每产出一个字就推给前端
    return collected_content        # 完整答案（但 for 循环拿不到！）


print_section("第 4 课：实战模拟 —— 流式输出 + 调用方自己拼接完整答案")

print("前端实时显示效果：", end="")
caller_buffer = ""
for piece in mock_stream_answer("你好，我是AI助手！"):
    print(piece, end="", flush=True)   # 模拟前端打字机效果
    caller_buffer += piece             # 调用方自己攒完整答案

print(f"\n调用方拼接出的完整答案：【{caller_buffer}】")
print(">>> 这就是实际项目中的正确姿势：依赖 yield 的值，不依赖 return。")



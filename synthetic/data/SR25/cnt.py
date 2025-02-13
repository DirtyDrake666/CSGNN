import networkx as nx

# 数据集文件名
SR25_NAMES = [
    'graph8c.g6',
    'sr16622.g6',
    'sr251256.g6',
    'sr261034.g6',
    'sr281264.g6',
    'sr291467.g6',
    'sr361446.g6',
    'sr401224.g6'
]

def count_cycles(file_name, max_cycle_length=None):
    """
    统计给定图文件中的简单环和无弦环数量。

    Args:
        file_name (str): 图文件名（Graph6 格式）。
        max_cycle_length (int): 最大环长度。

    Returns:
        dict: 包含简单环和无弦环数量的字典。
    """
    # 读取图文件
    Gs = nx.read_graph6(file_name)

    simple_cycle_count = 0
    chordless_cycle_count = 0

    for G in Gs:
        # 计算简单环
        simple_cycles = list(nx.simple_cycles(G, max_cycle_length))
        simple_cycle_count += len(simple_cycles)

        # 计算无弦环（Chordless cycles）
        if hasattr(nx, "chordless_cycles"):
            chordless_cycles = list(nx.chordless_cycles(G, max_cycle_length))
            chordless_cycle_count += len(chordless_cycles)
        else:
            print("NetworkX version does not support chordless_cycles.")

    return {
        "simple_cycles": simple_cycle_count,
        "chordless_cycles": chordless_cycle_count
    }

if __name__ == "__main__":
    max_cycle_length = 6

    # 遍历所有图文件
    for file_name in SR25_NAMES:
        print(f"Processing {file_name}...")
        try:
            result = count_cycles(file_name, max_cycle_length=max_cycle_length)
            print(f"File: {file_name}")
            print(f"  Simple Cycles (length <= {max_cycle_length}): {result['simple_cycles']}")
            print(f"  Chordless Cycles (length <= {max_cycle_length}): {result['chordless_cycles']}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

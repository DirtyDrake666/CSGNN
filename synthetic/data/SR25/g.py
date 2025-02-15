import tempfile
import networkx as nx

# 创建两个六元环图
G1 = nx.cycle_graph(6)
G2 = nx.cycle_graph(6)

# 创建一个空的图
G = nx.Graph()

# 将两个六元环图添加到空图中
G.add_nodes_from(G1.nodes(data=True))
G.add_edges_from(G1.edges())
G.add_nodes_from(G2.nodes(data=True))
mapping = {n: n + 6 for n in G2.nodes()}  # 将第二个六元环的节点编号加 6，以避免节点编号冲突
G.add_edges_from((mapping.get(u), mapping.get(v)) for u, v in G2.edges())


with tempfile.NamedTemporaryFile(delete=False) as f:
    nx.write_graph6(G, f.name)
    print("Temporary file path:", f.name)

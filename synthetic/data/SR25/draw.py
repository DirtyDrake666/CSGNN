import networkx as nx
import matplotlib.pyplot as plt

# 从 Graph6 文件中读取两个图
Gs=nx.read_graph6("sr16622.g6")

for G in Gs:
	nx.draw(G)
	plt.show()

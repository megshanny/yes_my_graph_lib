import networkx as nx
import time
import matplotlib.pyplot as plt
from itertools import combinations
from pyvis.network import Network
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from networkx.algorithms.flow import maximum_flow
from collections import deque
import heapq

class Graph:
    def __init__(self):
        self.graph = nx.Graph()
        self.node_map = {}  # 用于存储顶点映射
        self.reverse_map = {}  # 用于存储反向映射关系

    def add_edge(self, u, v):
        self.graph.add_edge(u, v)

    def remove_edge(self, u, v):
        self.graph.remove_edge(u, v)

    def nodes(self):
        return self.graph.nodes()

    def edges(self):
        return self.graph.edges()

    # 获取图的密度
    def density(self):
        return nx.density(self.graph)

    # 获取图的平均度
    def average_degree(self):
        degrees = dict(self.graph.degree()).values()
        return sum(degrees) / len(degrees) if len(degrees) > 0 else 0

    # 计算k-core分解并保存到文件
    def k_cores(self, output_path):
        start_time = time.time()
        cores = nx.core_number(self.graph)
        end_time = time.time()
        runtime = end_time - start_time

        try:
            with open(output_path, 'w') as f:
                f.write(f"{runtime:.4f}s\n")
                for node in sorted(cores):
                    original_node = self.reverse_map.get(node, node)
                    f.write(f"{original_node} {cores[node]}\n")
        except IOError as e:
            print(f"保存k-core分解结果时出现错误: {e}")
        except Exception as e:
            print(f"保存k-core分解结果时出现未知错误: {e}")

    def visualize(self, output_path, node_color='skyblue', node_size=500, edge_color='green', font_size=10, with_labels=True, layout='spring'):
        pos = self._get_layout(layout)
        
        # 创建图形
        plt.figure(figsize=(12, 12))
        # 优化布局
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=0.1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'shell':
            pos = nx.shell_layout(self.graph)
        elif layout == 'spectral':
            pos = nx.spectral_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)  # 默认使用 spring 布局
        # 节点颜色优化
        if isinstance(node_color, list):
            norm = Normalize(vmin=min(node_color), vmax=max(node_color))
            cmap = cm.viridis
            node_color = [cmap(norm(value)) for value in node_color]
        # 边颜色和宽度优化
        if isinstance(edge_color, list):
            norm = Normalize(vmin=min(edge_color), vmax=max(edge_color))
            cmap = cm.plasma
            edge_color = [cmap(norm(value)) for value in edge_color]
        
        edge_width = [self.graph[u][v].get('weight', 1.0) for u, v in self.graph.edges()]
        # 绘制图形
        nx.draw(self.graph, pos, with_labels=with_labels, node_size=node_size, node_color=node_color, edge_color=edge_color, width=edge_width, font_size=font_size)
        # 去除背景网格
        plt.axis('off')
        # 保存和展示图形
        plt.savefig(output_path)
        plt.show()

    def _get_layout(self, layout):
        if layout == 'spring':
            return nx.spring_layout(self.graph)
        elif layout == 'circular':
            return nx.circular_layout(self.graph)
        elif layout == 'random':
            return nx.random_layout(self.graph)
        elif layout == 'shell':
            return nx.shell_layout(self.graph)
        elif layout == 'spectral':
            return nx.spectral_layout(self.graph)
        else:
            raise ValueError(f"Unsupported layout: {layout}")

    # 交互式可视化图
    def visualize_interactive(self, output_file='graph.html', node_color='blue', node_size=10, edge_color='gray'):
        net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')
        net.barnes_hut()
        for node in self.graph.nodes:
            net.add_node(node, label=str(node), color=node_color, size=node_size)
        for edge in self.graph.edges:
            net.add_edge(edge[0], edge[1], color=edge_color)
        net.show_buttons(filter_=['physics'])
        net.save_graph(output_file)

    # def exact_densest_subgraph(self):
    #     def get_density(subgraph):
    #         nodes = subgraph.nodes()
    #         edges = subgraph.edges()
    #         return len(edges) / len(nodes)

    #     def binary_search_densest_subgraph():
    #         def check_density(graph, threshold):
    #             G = graph.copy()
    #             source, sink = 'source', 'sink'
    #             node_list = list(G.nodes())  # Copy the nodes list to avoid modifying during iteration
    #             for node in node_list:
    #                 G.add_edge(source, node, capacity=len(G.edges(node)))
    #                 G.add_edge(node, sink, capacity=threshold)
    #             flow_value, _ = maximum_flow(G, source, sink, flow_func=nx.algorithms.flow.edmonds_karp)
    #             return flow_value < len(G.edges)

    #         degrees = [deg for node, deg in self.graph.degree()]
    #         left, right = min(degrees), max(degrees)
    #         while right - left > 1e-5:
    #             mid = (left + right) / 2
    #             if check_density(self.graph, mid):
    #                 right = mid
    #             else:
    #                 left = mid

    #         return left

    #     max_density = binary_search_densest_subgraph()
    #     densest_subgraph = None

    #     for size in range(1, len(self.graph.nodes()) + 1):
    #         for subset in combinations(self.graph.nodes(), size):
    #             subgraph = self.graph.subgraph(subset)
    #             if get_density(subgraph) == max_density:
    #                 densest_subgraph = subgraph
    #                 break

    #     return densest_subgraph, max_density
    def build_flow_network(self, g):
        G = nx.DiGraph()
        S = 'source'
        T = 'sink'
        m = len(self.graph.edges)
        deg = dict(self.graph.degree())

        for u, v in self.graph.edges:
            G.add_edge(u, v, capacity=1)
            G.add_edge(v, u, capacity=0)  # Ensure reverse edge is added with 0 capacity

        for node in self.graph.nodes:
            G.add_edge(S, node, capacity=m)
            G.add_edge(node, T, capacity=m + 2 * g - deg[node])
            G.add_edge(T, node, capacity=0)  # Ensure reverse edge is added with 0 capacity
        return G, S, T
    
    def bfs(self, G, S, T):
        level = {node: -1 for node in G}
        level[S] = 0
        queue = deque([S])
        while queue:
            u = queue.popleft()
            for v in G[u]:
                if level[v] < 0 and G[u][v]['capacity'] > 0:
                    level[v] = level[u] + 1
                    queue.append(v)
        return level, level[T] >= 0

    def dfs(self, G, u, flow, T, level):
        if u == T:
            return flow
        total_flow = 0
        for v in G[u]:
            if level[v] == level[u] + 1 and G[u][v]['capacity'] > 0:
                min_cap = min(flow, G[u][v]['capacity'])
                pushed = self.dfs(G, v, min_cap, T, level)
                if pushed > 0:
                    G[u][v]['capacity'] -= pushed
                    if v not in G or u not in G[v]:
                        G.add_edge(v, u, capacity=0)
                    G[v][u]['capacity'] += pushed
                    total_flow += pushed
                    flow -= pushed
                    if flow == 0:
                        break
        return total_flow

    def dinic(self, g):
        G, S, T = self.build_flow_network(g)
        max_flow = 0
        while True:
            level, has_path = self.bfs(G, S, T)
            if not has_path:
                break
            flow = self.dfs(G, S, float('Inf'), T, level)
            while flow:
                max_flow += flow
                flow = self.dfs(G, S, float('Inf'), T, level)
        return max_flow

    def exact_densest_subgraph(self):
        left, right = 0, len(self.graph.edges)
        eps = 1e-5
        max_density = 0
        densest_subgraph = None

        while right - left > eps:
            mid = (left + right) / 2
            if self.dinic(mid) < len(self.graph.edges):
                right = mid
            else:
                left = mid

        G, S, T = self.build_flow_network(left)
        self.dinic(left)
        visited = set()
        queue = [S]

        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                for neighbor in G[node]:
                    if G[node][neighbor]['capacity'] > 0 and neighbor not in visited:
                        queue.append(neighbor)

        subgraph_nodes = [node for node in visited if node != S and node != T]
        densest_subgraph = self.graph.subgraph(subgraph_nodes)
        max_density = len(densest_subgraph.edges) / len(densest_subgraph.nodes)

        return densest_subgraph, max_density
    
    def greedy_densest_subgraph(self):
        graph = self.graph.copy()
        best_density = 0
        best_subgraph = graph.copy()

        while graph.number_of_nodes() > 0:
            density = nx.density(graph)
            if density > best_density:
                best_density = density
                best_subgraph = graph.copy()

            min_degree_node = min(graph.nodes, key=graph.degree)
            graph.remove_node(min_degree_node)

        return best_subgraph, best_density

    # 计算最密子图并保存到文件
    # 预处理以移除自环和重复边
    def preprocess(self):
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))  # 移除自环
        self.graph = nx.Graph(self.graph)  # 确保没有重复边

    def calculate_density(self, total_edges, remaining_nodes):
        if remaining_nodes == 0:
            return 0.0
        return total_edges / remaining_nodes
    
    def densest_subgraph(self, output_path, method="approx"):
        start_time = time.time()

        if method == "exact":
            subgraph, max_density = self.exact_densest_subgraph()
        elif method == "approx":
            subgraph, max_density = self.greedy_densest_subgraph()
        else:
            raise ValueError("Unsupported method. Use 'exact' or 'approx'.")

        end_time = time.time()
        runtime = end_time - start_time

        try:
            with open(output_path, 'w') as f:
                f.write(f"{runtime:.4f}s\n")
                f.write(f"density {max_density:.4f}\n")
                if subgraph is not None:
                    for node in sorted(subgraph.nodes()):
                        original_node = self.reverse_map.get(node, node)
                        f.write(f"{original_node} ")
                    f.write("\n")
        except IOError as e:
            print(f"保存最密子图结果时出现错误: {e}")
        except Exception as e:
            print(f"保存最密子图结果时出现未知错误: {e}")

    # def greedy_densest_subgraph(self):
    #     graph = self.graph
    #     n = graph.number_of_nodes()
    #     degrees = dict(graph.degree())
    #     total_edges = graph.number_of_edges()
    #     total_edges *= 2  # 每条边在无向图中被计算了两次

    #     max_density = 0.0
    #     densest_subgraph = []
    #     remaining_nodes = n

    #     min_heap = [(degree, node) for node, degree in degrees.items()]
    #     heapq.heapify(min_heap)
    #     moved = {node: False for node in graph.nodes()}

    #     while remaining_nodes > 0:
    #         while min_heap:
    #             degree, v = heapq.heappop(min_heap)
    #             if not moved[v]:
    #                 break

    #         if moved[v]:
    #             break

    #         moved[v] = True
    #         total_edges -= degrees[v]
    #         remaining_nodes -= 1

    #         for u in list(graph.neighbors(v)):
    #             if not moved[u]:
    #                 degrees[u] -= 1
    #                 heapq.heappush(min_heap, (degrees[u], u))

    #         current_density = self.calculate_density(total_edges, remaining_nodes)
    #         if current_density > max_density:
    #             max_density = current_density
    #             densest_subgraph = [node for node in graph.nodes() if not moved[node]]

    #     subgraph = graph.subgraph(densest_subgraph).copy()
    #     return subgraph, max_density

    # k-clique分解
    def bron_kerbosch(self, r, p, x, cliques):
        if not p and not x:
            cliques.append(r)
            return
        for v in list(p):
            self.bron_kerbosch(r | {v}, p & set(self.graph.neighbors(v)), x & set(self.graph.neighbors(v)), cliques)
            p.remove(v)
            x.add(v)
    def find_maximal_cliques(self):
        cliques = []
        self.bron_kerbosch(set(), set(self.graph.nodes()), set(), cliques)
        return cliques
    def k_clique_decomposition(self, k):
        maximal_cliques = self.find_maximal_cliques()
        k_cliques = [clique for clique in maximal_cliques if len(clique) == k]
        return k_cliques

    # 使用启发式算法计算k-clique最密子图
    def k_clique_densest_subgraph(self, k, iterations=1000):
        r = {node: 0 for node in self.graph.nodes}
        for _ in range(iterations):
            s = r.copy()
            for clique in self._find_k_cliques(k):
                min_node = min(clique, key=lambda node: s[node])
                r[min_node] += 1
        for node in r:
            r[node] /= iterations
        densest_subgraph, density = self._extract_densest_subgraph(r, k)
        return densest_subgraph, density
    def _find_k_cliques(self, k):
        # 使用networkx查找所有k-cliques
        cliques = [clique for clique in nx.find_cliques(self.graph) if len(clique) == k]
        return cliques
    def _extract_densest_subgraph(self, r, k):
        sorted_nodes = sorted(r, key=r.get, reverse=True)
        subgraph_nodes = sorted_nodes[:k]
        subgraph = self.graph.subgraph(subgraph_nodes)
        return subgraph, nx.density(subgraph)
    # 查找并保存k-clique最密子图
    def find_k_clique_densest_subgraph(self, k, output_path, iterations=1000):
        start_time = time.time()
        subgraph, density = self.k_clique_densest_subgraph(k, iterations)
        end_time = time.time()
        runtime = end_time - start_time
        try:
            with open(output_path, 'w') as f:
                f.write(f"{runtime:.4f}s\n")
                f.write(f"{density:.4f}\n")
                for node in sorted(subgraph.nodes()):
                    original_node = self.reverse_map.get(node, node)
                    f.write(f"{original_node} ")
                f.write("\n")
        except IOError as e:
            print(f"保存k-clique最密子图结果时出现错误: {e}")
        except Exception as e:
            print(f"保存k-clique最密子图结果时出现未知错误: {e}")

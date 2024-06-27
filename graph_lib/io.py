from .graph import Graph

def load_graph(file_path):
    graph = Graph()
    node_map = {}
    reverse_map = {}
    current_index = 0

    try:
        with open(file_path, 'r') as f:
            # 读取顶点和边的数量
            first_line = f.readline().strip().split()
            if len(first_line) != 2:
                # 第二个参数是顶点的数量，第四个参数是边的数量
                n, m = first_line[1], first_line[3]
                # 跳过第二行
                f.readline()
        
            else:
                n, m = map(int, first_line) 

            for line in f:
                try:
                    u, v = map(int, line.strip().split())

                    # 处理重边和自环
                    if u == v:
                        continue

                    if (u, v) not in graph.edges() and (v, u) not in graph.edges():
                        if u not in node_map:
                            node_map[u] = current_index
                            reverse_map[current_index] = u
                            current_index += 1

                        if v not in node_map:
                            node_map[v] = current_index
                            reverse_map[current_index] = v
                            current_index += 1

                        graph.add_edge(node_map[u], node_map[v])
                except ValueError:
                    print(f"跳过错误的行: {line.strip()}")

        graph.node_map = node_map
        graph.reverse_map = reverse_map
    except FileNotFoundError:
        print(f"未找到文件: {file_path}")
    except ValueError as e:
        print(f"文件格式错误: {e}")
    except Exception as e:
        print(f"加载图时出现错误: {e}")

    return graph

def save_graph(graph, file_path):
    try:
        with open(file_path, 'w') as f:
            f.write(f"{len(graph.nodes())} {len(graph.edges())}\n")
            for u, v in graph.edges():
                original_u = graph.reverse_map[u]
                original_v = graph.reverse_map[v]
                f.write(f"{original_u} {original_v}\n")
    except IOError as e:
        print(f"保存图时出现错误: {e}")
    except Exception as e:
        print(f"保存图时出现未知错误: {e}")

import argparse
import os
import datetime
from graph_lib import Graph, load_graph, save_graph

def get_output_path(output_dir, input_file, prefix, extension):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_name = os.path.basename(input_file)
    name, _ = os.path.splitext(base_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return os.path.join(output_dir, f"{name}_{prefix}_{timestamp}.{extension}")

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Graph Analysis Tool")
    parser.add_argument('input_file', type=str, help='Path to the input graph file')
    parser.add_argument('--save', action='store_true', help='Save the graph')
    parser.add_argument('--density', action='store_true', help='Calculate the density of the graph')
    parser.add_argument('--average_degree', action='store_true', help='Calculate the average degree of the graph')
    parser.add_argument('--k_cores', action='store_true', help='Compute and save k-core decomposition')
    parser.add_argument('--densest_subgraph', choices=['exact', 'approx'], help='Compute and save densest subgraph (choose exact or approx method)')
    parser.add_argument('--visualize', action='store_true', help='Visualize the graph statically')
    parser.add_argument('--visualize_interactive', action='store_true', help='Visualize the graph interactively')
    parser.add_argument('--k_clique_densest', type=int, help='Find and save densest k-clique subgraph, provide k')
    parser.add_argument('--bron_kerbosch_cliques', type=int, help='Find and save maximal k-cliques using Bron-Kerbosch algorithm, provide k')
    parser.add_argument('--layout', type=str, choices=['spring', 'circular', 'random', 'shell', 'spectral'], default='spring', help='Layout for static visualization')
    parser.add_argument('--node_color', type=str, default='skyblue', help='Node color for visualization')
    parser.add_argument('--node_size', type=int, default=500, help='Node size for visualization')
    parser.add_argument('--edge_color', type=str, default='black', help='Edge color for visualization')
    parser.add_argument('--font_size', type=int, default=10, help='Font size for visualization')
    args = parser.parse_args()

    # 加载图
    g = load_graph(args.input_file)
    if not g.nodes():  # 如果加载失败，返回的图中没有节点
        print("加载图失败，请检查输入文件格式和路径")
        return

    # 保存图
    if args.save:
        output_path = get_output_path('outputs/graph', args.input_file, 'graph', 'txt')
        save_graph(g, output_path)
        print(f"图已保存到: {output_path}")

    # 计算图的密度
    if args.density:
        density = g.density()
        print(f"Graph Density: {density}")

    # 计算图的平均度
    if args.average_degree:
        average_degree = g.average_degree()
        print(f"Average Degree: {average_degree}")

    # 计算k-core分解并保存结果
    if args.k_cores:
        output_path = get_output_path('outputs/k_core', args.input_file, 'k_core', 'txt')
        g.k_cores(output_path)
        print(f"k-core分解结果已保存到: {output_path}")

    # 计算最密子图并保存结果
    if args.densest_subgraph:
        method = args.densest_subgraph
        output_path = get_output_path(f'outputs/densest_subgraph_{method}', args.input_file, f'densest_subgraph_{method}', 'txt')
        g.densest_subgraph(output_path, method)
        print(f"最密子图结果已保存到: {output_path}")

    # 静态可视化图
    if args.visualize:
        output_path = get_output_path('outputs/visualization', args.input_file, 'visualization', 'png')
        g.visualize(output_path, node_color=args.node_color, node_size=args.node_size, edge_color=args.edge_color, font_size=args.font_size, layout=args.layout)
        print(f"图的静态可视化结果已保存到: {output_path}")

    # 交互式可视化图
    if args.visualize_interactive:
        output_path = get_output_path('outputs/visualization_interactive', args.input_file, 'visualization_interactive', 'html')
        g.visualize_interactive(output_path, node_color=args.node_color, node_size=args.node_size, edge_color=args.edge_color)
        print(f"图的交互式可视化结果已保存到: {output_path}")

    # 分解cliques
    if args.bron_kerbosch_cliques:
        k = args.bron_kerbosch_cliques
        cliques = g.k_clique_decomposition(k)
        output_path = get_output_path('outputs/bron_kerbosch_cliques', args.input_file, 'bron_kerbosch_cliques', 'txt')
        with open(output_path, 'w') as f:
            for clique in cliques:
                f.write(f"{clique}\n")
        print(f"最大k-cliques (k={k}): {cliques}")
        print(f"Bron-Kerbosch最大cliques结果已保存到: {output_path}")

    # 查找并保存k-clique最密子图结果
    if args.k_clique_densest:
        k = args.k_clique_densest
        output_path = get_output_path('outputs/k_clique_densest', args.input_file, f'k_clique_densest_{k}', 'txt')
        g.find_k_clique_densest_subgraph(k, output_path)
        print(f"k-clique最密子图结果已保存到: {output_path}")

if __name__ == "__main__":
    main()

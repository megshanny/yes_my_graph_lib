def calculate_density(graph):
    return graph.density()

def calculate_average_degree(graph):
    return graph.average_degree()

def k_cores(graph, output_path):
    return graph.k_cores(output_path)

def densest_subgraph(graph, output_path, method):
    return graph.densest_subgraph(output_path, method)

def visualize(graph, output_path, **kwargs):
    return graph.visualize(output_path, **kwargs)

def visualize_interactive(graph, output_path, **kwargs):
    return graph.visualize_interactive(output_path, **kwargs)

def dynamic_k_core(graph, k, operations):
    return graph.dynamic_k_core(k, operations)

def k_clique_decomposition(graph, k):
    return graph.k_clique_decomposition(k)

def k_clique_densest_subgraph(graph, k):
    return graph.k_clique_densest_subgraph(k)



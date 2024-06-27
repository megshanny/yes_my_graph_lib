### README

## Graph Analysis Tool

This project aims to implement and visualize several common algorithms in complex networks, including basic graph metrics, densest subgraph, k-core decomposition, k-clique densest subgraph, and more. The project provides both static and interactive visualizations, supporting custom styles for nodes and edges, as well as interactive features such as zooming, panning, and layout adjustments.

### Project Structure

```plaintext
LAB/
│
├── data/                                # Folder for storing graph data files
│
├── graph_lib/                           # Core library implementing graph algorithms
│   ├── __init__.py                      # Initialization file
│   ├── algorithms.py                    # Implementation of various graph algorithms
│   ├── graph.py                         # Basic graph operations and algorithm implementation
│   ├── io.py                            # Graph input/output functionality
│   └── visualization.py                 # Graph visualization functionality
│
├── lib/                                 # Additional library files
│
├── outputs/                             # Folder for storing output results
│   ├── bron_kerbosch_cliques/           # Output results of Bron-Kerbosch algorithm
│   ├── densest_subgraph_approx/         # Output results of greedy densest subgraph algorithm
│   ├── densest_subgraph_exact/          # Output results of exact densest subgraph algorithm
│   ├── graph/                           # Output results of basic graph operations
│   ├── k_clique_densest/                # Output results of k-clique densest subgraph
│   ├── k_core/                          # Output results of k-core decomposition
│   ├── visualization/                   # Output results of static visualization
│   └── visualization_interactive/       # Output results of interactive visualization
│
├── install.ipynb                        # Jupyter Notebook for installing dependencies
├── main.py                              # Main program file
└── README.md                            # Project introduction and usage guide
```

### Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/megshanny/yes_my_graph_lib.git
   cd yes_my_graph_lib/LAB
   ```

2. **Install dependencies**:
   Open `install.ipynb` and run the cells to install the required Python packages, or manually install them using pip:
   ```sh
   pip install -r requirements.txt
   ```

### Usage

#### Command-line Interface

The main program file `main.py` provides a command-line interface to run various graph algorithms and visualizations. Below are examples of how to use the tool.

1. **Calculate and save basic graph metrics**:
   ```sh
   python main.py data/your_graph.txt --density --average_degree
   ```

2. **Calculate and save k-core decomposition**:
   ```sh
   python main.py data/your_graph.txt --k_cores
   ```

3. **Calculate and save densest subgraph (exact algorithm)**:
   ```sh
   python main.py data/your_graph.txt --densest_subgraph exact
   ```

4. **Generate and save static visualization**:
   ```sh
   python main.py data/your_graph.txt --visualize --layout circular --node_color red --node_size 600 --edge_color blue --font_size 12
   ```

5. **Generate and save interactive visualization**:
   ```sh
   python main.py data/your_graph.txt --visualize_interactive --node_color green --node_size 15 --edge_color gray
   ```

6. **Find and save maximal k-cliques using Bron-Kerbosch algorithm**:
   ```sh
   python main.py data/your_graph.txt --bron_kerbosch_cliques 4
   ```

7. **Find and save densest k-clique subgraph**:
   ```sh
   python main.py data/your_graph.txt --k_clique_densest 4
   ```

#### Example Commands

- **Calculate graph density**:
  ```sh
  python main.py data/sample_graph.txt --density
  ```

- **Calculate average degree of the graph**:
  ```sh
  python main.py data/sample_graph.txt --average_degree
  ```

- **Save the graph**:
  ```sh
  python main.py data/sample_graph.txt --save
  ```

- **Calculate and save k-core decomposition**:
  ```sh
  python main.py data/sample_graph.txt --k_cores
  ```

- **Calculate and save the exact densest subgraph**:
  ```sh
  python main.py data/sample_graph.txt --densest_subgraph exact
  ```

- **Generate and save static visualization with specific layout and styles**:
  ```sh
  python main.py data/sample_graph.txt --visualize --layout spring --node_color blue --node_size 700 --edge_color red --font_size 14
  ```

- **Generate and save interactive visualization**:
  ```sh
  python main.py data/sample_graph.txt --visualize_interactive --node_color yellow --node_size 20 --edge_color green
  ```

- **Find and save maximal 4-cliques using Bron-Kerbosch algorithm**:
  ```sh
  python main.py data/sample_graph.txt --bron_kerbosch_cliques 4
  ```

- **Find and save densest 4-clique subgraph**:
  ```sh
  python main.py data/sample_graph.txt --k_clique_densest 4
  ```

### Features

- **Basic Graph Operations**: Load graphs from files, add/remove edges, and save graphs.
- **Graph Metrics**: Calculate graph density and average degree.
- **k-core Decomposition**: Compute and save k-core decomposition.
- **Densest Subgraph**: Compute and save the densest subgraph using exact and approximate algorithms.
- **k-clique Decomposition**: Find and save maximal k-cliques using the Bron-Kerbosch algorithm.
- **k-clique Densest Subgraph**: Compute and save the densest k-clique subgraph using a heuristic algorithm.
- **Visualization**: Generate and save static and interactive visualizations with customizable styles.

### Contributing

Contributions are welcome! If you have any ideas or suggestions, feel free to create an issue or submit a pull request.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
(not yet, take it easy)

## Warning

I'm sorry to inform you that the accuracy of this is not guaranteed because the teaching assistant has not graded it yet.
import networkx as nx
import matplotlib.pyplot as plt

def plot_grahp(G):
    # Draw the graph
    # nx.draw(G, with_labels=True, node_color='skyblue', node_size=700, font_size=16, font_color='black')
    nx.draw(
        G,
        pos=nx.circular_layout(G),
        with_labels=True,
        node_color="skyblue",
    )
    plt.show()


def md_sequence():
    # Create a graph object
    G = nx.Graph()

    # Add edges
    G.add_edge((0,1), (0,2))
    G.add_edge((0,1), (0,3))
    G.add_edge((0,2), (1,1))
    G.add_edge((0,2), (1,4))
    G.add_edge((0,3), (1,5))
    G.add_edge((0,3), (1,2))
    G.add_edge((1,4), (1,3))
    G.add_edge((1,5), (1,3))


    H = nx.Graph()
    H.add_node((1,3))

    # intersection
    I = nx.intersection(G,H)

    # difference
    D = G.copy()
    D.remove_nodes_from(n for n in G.nodes if n in I)

    # same co_dimension coupling
    C = nx.Graph()
    C.add_edge((1,3), (1,4))
    C.add_edge((1,3), (1,5))


    # union
    U = D.copy()
    U.add_edges_from(e for e in C.edges)

    equal = nx.utils.graphs_equal(U, G)
    plot_grahp(G)

def md_conformal_to_no_conformal():
    # Create a graph object
    G = nx.DiGraph()
    G.add_edge((0,1), (0,2))
    G.add_edge((0,1), (0,3))
    G.add_edge((0,2), (1,1))
    G.add_edge((0,2), (1,3))
    G.add_edge((0,3), (1,3))
    G.add_edge((0,3), (1,2))

    # disjoint subdomain
    H = nx.DiGraph()
    H.add_node((1,3))

    # 1) Compute intersection
    I = nx.intersection(G,H)
    # 2) same co_dimension coupling
    C = nx.DiGraph()
    C.add_edge((1,3), (1,4))
    C.add_edge((1,3), (1,5))
    # 3) add extra duplicates
    G.add_edge((0,2), (1,4))
    G.add_edge((0,3), (1,5))

    # Difference cut conformity contained in I
    D = G.copy()
    D.remove_nodes_from(n for n in G.nodes if n in I)

    # union: md non-conformal
    U = D.copy()
    U.add_edges_from(e for e in C.edges)

    cycles = list(nx.chordless_cycles(U))
    plot_grahp(U)

# md_sequence()

md_conformal_to_no_conformal()
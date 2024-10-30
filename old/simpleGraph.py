import networkx as nx
import matplotlib.pyplot as plt
import random
from simpleNode import SimpleNode as Node

G = nx.Graph()
#the node number seems to just be an identifier

#each argumetn in add_node is an identifier
# G.add_node(1)
# G.add_nodes_from([(2, {"color":"red"}),3])
# G.add_edge(1,2)
# G.add_node("bob")
# G.add_node("joe")
# G.add_edge("bob","joe")
# G.add_edge("bob", 1)

#can store a node object in the graph
#... or 10
n = [Node(0.5) for i in range(10)]
G.add_nodes_from(n)

#colors work properly
activeNode = Node(0.5)
activeNode.active = True
G.add_node(activeNode)

print(G.nodes())
print(G.nodes().items())

node_color_map = []
for node in G:
    if node.active == True:
        node_color_map.append('green')
    else:
        node_color_map.append('red')

#creates random edges
edges = set()
while len(edges) < 30:
    # Pick two random nodes and ensure they are different
    node1, node2 = random.sample(list(G.nodes()), 2)
    edge = (node1, node2)
    
    # Add edge to the set (automatically avoids duplicates)
    edges.add(edge)

# Add the random edges to the graph
G.add_edges_from(edges)

#set activation based on live/dead nodes
for edge in G.edges():
    node1, node2 = edge
    # If either node is active, the edge is active
    G.edges[edge]['active'] = (node1.isActive() or node2.isActive())

edge_color_map = ['blue' if G.edges[edge]['active'] else 'gray' for edge in G.edges()]




nx.draw(G, with_labels=False, node_color=node_color_map, edge_color=edge_color_map, node_size=800, font_color='white')
print("AAA")

plt.show() 


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from network_tops import make_sw_net

num_nodes=20
k_val = 4
p_rewire = 0.1
net = make_sw_net(num_nodes, k_val, p_rewire)
print("Generated Network Matrix:")
print(net)

graph = nx.Graph(net)
plt.figure(figsize=(8,8))

pos = nx.spring_layout(graph, seed=42)

nx.draw(graph, pos, with_labels=False, node_color='skyblue', node_size=300,
                 edge_color='gray', linewidths=1, font_size=10, alpha=0.9)

plt.title(f"Node-Link diagram for Network (N={num_nodes}, K={k_val}, Rewire prob ={p_rewire})")
plt.axis('off')
plt.show()



def generate_fibonacci_sphere_points(num_points, radius=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    points = []
    phi = np.pi * (3. - np.sqrt(5.))

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2
        r = np.sqrt(1 - y * y)
        theta = phi * i

        x = np.cos(theta) * r
        z = np.sin(theta) * r

        points.append(np.array([x, y, z]) * radius)
    return {i: p for i, p in enumerate(points)}

def get_great_circle_arc(p1, p2, radius, num_segments=20):
    """
    Calculates points along the great circle arc between two 3D points on a sphere.

    Args:
        p1 (np.array): 3D coordinates of the first point.
        p2 (np.array): 3D coordinates of the second point.
        radius (float): The radius of the sphere.
        num_segments (int): Number of intermediate points for the arc.

    Returns:
        tuple: (x_coords, y_coords, z_coords) for the arc.
    """
    p1_unit = p1 / np.linalg.norm(p1)
    p2_unit = p2 / np.linalg.norm(p2)

    dot_product = np.dot(p1_unit, p2_unit)
    dot_product = np.clip(dot_product, -1.0, 1.0) 
    theta = np.arccos(dot_product)

    arc_points = []
    if theta < 1e-6:
        arc_points = [p1, p2]
    elif np.pi - theta < 1e-6:
        if np.abs(p1_unit[2]) < 0.9:
            p_perp = np.array([-p1_unit[1], p1_unit[0], 0])
        else:
            p_perp = np.array([0, -p1_unit[2], p1_unit[1]])
        p_perp = p_perp / np.linalg.norm(p_perp)
        
        for t in np.linspace(0, 1, num_segments):
            if t < 0.5:
                sub_t = t * 2
                sub_theta = np.pi / 2
                point = (np.sin((1-sub_t)*sub_theta) / np.sin(sub_theta)) * p1_unit + \
                        (np.sin(sub_t)*sub_theta / np.sin(sub_theta)) * p_perp
            else:
                sub_t = (t - 0.5) * 2
                sub_theta = np.pi / 2
                point = (np.sin((1-sub_t)*sub_theta) / np.sin(sub_theta)) * p_perp + \
                        (np.sin(sub_t)*sub_theta / np.sin(sub_theta)) * (-p1_unit)
            arc_points.append(point * radius)
    else:
        for t in np.linspace(0, 1, num_segments):
            point = (np.sin((1-t)*theta) / np.sin(theta)) * p1_unit + \
                    (np.sin(t*theta) / np.sin(theta)) * p2_unit
            arc_points.append(point * radius)
            
    arc_points = np.array(arc_points)
    return arc_points[:, 0], arc_points[:, 1], arc_points[:, 2]

num_nodes_main = 20
k_neighbors_main = 4
p_rewire_main = 0.1
sphere_radius_main = 1.0
spring_layout_iterations_main = 10
num_arc_segments_main = 20

np_random_seed_main = 42
random_seed_main = 42
np.random.seed(np_random_seed_main)
random.seed(random_seed_main)

adj_matrix = make_sw_net(num_nodes_main, k_neighbors_main, p_rewire_main)
G_watts_strogatz = nx.Graph(adj_matrix)
print(f"Watts-Strogatz graph generated using make_sw_net. Edges: {G_watts_strogatz.number_of_edges()}")

print("\nAdjacency Matrix of the generated Watts-Strogatz network:")
print(nx.to_numpy_array(G_watts_strogatz, dtype=int))

initial_sphere_pos = generate_fibonacci_sphere_points(num_nodes_main, radius=sphere_radius_main, seed=np_random_seed_main)

spring_layout_pos = nx.spring_layout(G_watts_strogatz, dim=3, pos=initial_sphere_pos,
                                    iterations=spring_layout_iterations_main, seed=np_random_seed_main)

final_topology_aware_pos = {}
for node, coords in spring_layout_pos.items():
    norm = np.linalg.norm(coords)
    if norm == 0:
        final_topology_aware_pos[node] = np.array([sphere_radius_main, 0, 0])
    else:
        final_topology_aware_pos[node] = (coords / norm) * sphere_radius_main

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

core_radius_display = sphere_radius_main * 0.98 
u_core = np.linspace(0, 2 * np.pi, 50)
v_core = np.linspace(0, np.pi, 50)
x_core = core_radius_display * np.outer(np.cos(u_core), np.sin(v_core))
y_core = core_radius_display * np.outer(np.sin(u_core), np.sin(v_core))
z_core = core_radius_display * np.outer(np.ones(np.size(u_core)), np.cos(v_core))
ax.plot_surface(x_core, y_core, z_core, color='lightgray', alpha=0.3, linewidth=0, zorder=0) 


for edge in G_watts_strogatz.edges():
    node1, node2 = edge
    p1 = final_topology_aware_pos[node1]
    p2 = final_topology_aware_pos[node2]
    
    x_arc, y_arc, z_arc = get_great_circle_arc(p1, p2, sphere_radius_main, num_segments=num_arc_segments_main)
    
    ax.plot(x_arc, y_arc, z_arc, c='white', alpha=1, linewidth=3, zorder=1)

x_nodes = [final_topology_aware_pos[node][0] for node in G_watts_strogatz.nodes()]
y_nodes = [final_topology_aware_pos[node][1] for node in G_watts_strogatz.nodes()]
z_nodes = [final_topology_aware_pos[node][2] for node in G_watts_strogatz.nodes()]

ax.scatter(x_nodes, y_nodes, z_nodes, c='salmon', s=100, alpha=1.0, edgecolor='black', zorder=2)

ax.set_title(
    f"3D Watts-Strogatz Network (N={num_nodes_main}, K={k_neighbors_main}, p={p_rewire_main})\n"
    f"with Surface Edges", fontsize=12
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_box_aspect([1,1,1])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_facecolor('black')

plt.show()
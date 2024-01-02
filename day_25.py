import random

import itertools
from collections import defaultdict
from copy import copy, deepcopy

from tqdm import tqdm
import networkx as nx

from notebooks.aoc_2023.utils import load_stripped_lines


def dfs(start, edges):
    visited = {start}
    predecessors = {start: None}
    open = [start]

    while open:
        current = open.pop()

        # if current == goal:
        #     break

        for adjacent in edges[current]:
            if adjacent not in visited:
                visited.add(adjacent)
                open.append(adjacent)
                predecessors[adjacent] = current

    return visited


def compute_component(vertices, edges):
    components = []
    open_states: set = copy(vertices)
    while open_states:
        vertex = open_states.pop()
        connected = dfs(vertex, edges)
        components.append(connected)
        open_states.difference_update(connected)
    return components


def parse_graph(data: list[str]):
    counter = itertools.count()
    map_string_to_int: dict[str, int] = {}
    vertices: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for line in data:
        vertex = line.split(':')[0]
        if vertex not in map_string_to_int:
            vertex_id = next(counter)
            map_string_to_int[vertex] = vertex_id
        else:
            vertex_id = map_string_to_int[vertex]

        connected_to = line.split(': ')[1].split(' ')

        for neighbour in connected_to:
            if neighbour in map_string_to_int:
                neighbour_id = map_string_to_int.get(neighbour)
            else:
                neighbour_id = next(counter)
                map_string_to_int[neighbour] = neighbour_id
            if vertex_id < neighbour_id:
                edges.add((vertex_id, neighbour_id))
            else:
                edges.add((neighbour_id, vertex_id))
    id_of_interest = [721, 794, 229, 701, 502, 1194]
    for key, value in map_string_to_int.items():
        if value in id_of_interest:
            print(value, key)


    G = nx.Graph()
    G.add_edges_from(edges)
    return G, map_string_to_int


def compute_unique_edges(edges) -> set[tuple[str, str]]:
    unique_edges = set()
    for key, values in edges.items():
        for val in values:
            if key < val:
                unique_edges.add((key, val))

    len_edges = sum(len(values) for values in edges.values())
    len_unique_edges = len(unique_edges)
    assert len_unique_edges * 2 == len_edges
    return unique_edges


def remove_selected_edges(edges, edge_1, edge_2, edge_3):
    new_edges = deepcopy(edges)
    for edge in (edge_1, edge_2, edge_3):
        new_edges[edge[0]].remove(edge[1])
        new_edges[edge[1]].remove(edge[0])
    return new_edges


def solve_1(vertices, edges):
    unique_edges = compute_unique_edges(edges)
    print(f'len: {len(unique_edges)}')
    for edge_1, edge_2, edge_3 in tqdm(itertools.combinations(unique_edges, 3)):
        '''hfx/pzl, the wire between bvb/cmg, and the wire between nvd/jqt,'''
        #edge_1 = ('hfx', 'pzl')
        #edge_2 = ('bvb', 'cmg')
        #edge_3 = ('jqt', 'nvd')
        edges_removed = remove_selected_edges(edges, edge_1, edge_2, edge_3)
        components = compute_component(vertices, edges_removed)
        if len(components) == 2:
            solution = len(components[0]) * len(components[1])
            return solution


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    # Search function

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    #  Applying Kruskal algorithm
    def kruskal_algo(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.apply_union(parent, rank, x, y)
        for u, v, weight in result:
            print("%d - %d: %d" % (u, v, weight))
        return result


def find_edge_cut(graph):
    for node_1, node_2 in itertools.combinations(graph.nodes(), 2):
        cut_value = None
        try:
            cut_value, partition = nx.minimum_cut(graph, node_1, node_2)
        except Exception as e:
            print(e)
            continue

        if cut_value is not None:
            if cut_value == 3:
                # You've found a cut of size 3
                cut_edges = [(u, v) for u, v in graph.edges() if u in partition[0] and v in partition[1]]
                return cut_edges


def karger_min_cut(graph, _map):
    while len(graph.nodes()) > 2:
        edge = random.choice(list(graph.edges()))
        graph = nx.contracted_edge(graph, edge, self_loops=False)
        pass
    a = list(graph.edges())
    return a

if __name__ == '__main__':
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/25.txt')
    g, _map = parse_graph(raw_data)
    karger_min_cut(g, _map)
    find_edge_cut(g)
    nx.draw(g, with_labels=True)
    g.edges()
    pass


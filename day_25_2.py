import random

import itertools
from collections import defaultdict
from copy import copy, deepcopy

from tqdm import tqdm

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
    vertices: set[str] = set()
    edges: dict[str, set[str]] = defaultdict(set)
    for line in data:
        vertex = line.split(':')[0]
        connected_to = line.split(': ')[1].split(' ')
        vertices.add(vertex)
        edges[vertex].update(connected_to)

        for neighbour in connected_to:
            vertices.add(neighbour)
            edges[neighbour].add(vertex)

    return vertices, edges


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
    for _, _, _ in tqdm(itertools.combinations(unique_edges, 3)):
        '''hfx/pzl, the wire between bvb/cmg, and the wire between nvd/jqt,'''
        edge_1 = ('pbq', 'nzn')
        edge_2 = ('vfs', 'dhl')
        edge_3 = ('xvp', 'zpc')
        edges_removed = remove_selected_edges(edges, edge_1, edge_2, edge_3)
        components = compute_component(vertices, edges_removed)
        if len(components) == 2:
            solution = len(components[0]) * len(components[1])
            return solution


if __name__ == '__main__':
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/25.txt')
    _graph = parse_graph(raw_data)
    s1 = solve_1(*_graph)
    print(s1)
    #nb_compontn = number_of_component(*_graph)


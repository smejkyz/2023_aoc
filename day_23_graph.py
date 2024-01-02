import heapq
from copy import deepcopy

import math
from collections import defaultdict

import itertools

import numpy as np
from tqdm import tqdm

from notebooks.aoc_2023.day_17_mult import Direction
from notebooks.aoc_2023.utils import load_stripped_lines, Grid, COORDINATE, PrioritizedState, _reconstruct_path, reconstruct_path


def vis_path(path: list[COORDINATE], height: int, width: int) -> np.ndarray:
    vis = np.full(shape=(height, width), fill_value='.', dtype='<U8')
    for i, item in enumerate(path):
        vis[item] = i
    return vis


def get_direction(p_1, p_2) -> Direction:
    diff = p_2[0] - p_1[0], p_2[1] - p_1[1]
    if diff == (0, 1):
        return Direction.RIGHT
    if diff == (0, -1):
        return Direction.LEFT
    if diff == (1, 0):
        return Direction.DOWN
    if diff == (-1, 0):
        return Direction.UP


def is_consistent(current: COORDINATE, next: COORDINATE, symbol: str) -> bool:
    direction = get_direction(current, next)
    if direction == Direction.DOWN and symbol == 'v':
        return True
    if direction == Direction.UP and symbol == '^':
        return True
    if direction == Direction.RIGHT and symbol == '>':
        return True
    if direction == Direction.LEFT and symbol == '<':
        return True
    return False

PATHS = []

MAXIMUM = 0

def find_paths(grid, path_so_far: list[COORDINATE], end: COORDINATE) -> bool:
    global MAXIMUM
    while True:
        if path_so_far[-1] == end:
            PATHS.append(path_so_far)
            if (len(path_so_far) - 1) > MAXIMUM:
                MAXIMUM = len(path_so_far) - 1
                print(f'new maximum found: {MAXIMUM}')
            return True

        next_steps = [neighbour for neighbour in grid.four_neighbours(path_so_far[-1]) if neighbour not in path_so_far and grid[neighbour] != '#']

        next_steps_corrected = []
        for next_step in next_steps:
            if grid[next_step] in ('^', '>', 'v', '<'):
                if is_consistent(path_so_far[-1], next_step, grid[next_step]):
                    next_steps_corrected.append(next_step)
            else:
                next_steps_corrected.append(next_step)

        if len(next_steps_corrected) > 1 and grid[path_so_far[-1]] in ('^', '>', 'v', '<'):
            # todo
            raise NotImplemented()
        elif len(next_steps_corrected) > 1:
            break
        elif len(next_steps_corrected) == 0:
            # dead end:
            return False
        else:
            path_so_far.append(next_steps_corrected[0])
            # path_vis = vis_path(path_so_far, grid.height, grid.width)
    # I am on intersection
    for next_step in next_steps_corrected:
        find_paths(grid, path_so_far + [next_step], end)


def solve_1(grid: Grid) -> int:
    start_coordinate = [coor for coor in grid.get_coordinates('.') if coor[0] == 0][0]
    end_coordinate = [coor for coor in grid.get_coordinates('.') if coor[0] == grid.height -1][0]

    find_paths(grid, [start_coordinate], end_coordinate)

    paths_steps = sorted([len(path) - 1 for path in PATHS])
    return max(paths_steps)


def solve_2(grid: Grid) -> int:
    # treat extra symbols as normal symbols
    for symbol in ('^', '>', 'v', '<'):
        for pos in grid.get_coordinates(symbol):
            grid[pos] = '.'
    start_coordinate = [coor for coor in grid.get_coordinates('.') if coor[0] == 0][0]
    end_coordinate = [coor for coor in grid.get_coordinates('.') if coor[0] == grid.height -1][0]

    find_paths(grid, [start_coordinate], end_coordinate)

    paths_steps = sorted([len(path) - 1 for path in PATHS])
    return max(paths_steps)


def parse_graph(grid: Grid):
    start_coordinate = [coor for coor in grid.get_coordinates('.') if coor[0] == 0][0]
    end_coordinate = [coor for coor in grid.get_coordinates('.') if coor[0] == grid.height - 1][0]
    vertices = [start_coordinate, end_coordinate]

    for coor in itertools.product(range(grid.height), range(grid.width)):
        if grid[coor] == '#':
            continue
        next_steps = [neighbour for neighbour in grid.four_neighbours(coor) if grid[neighbour] != '#']
        if len(next_steps) > 2:
            vertices.append(coor)

    # find ednges for each vertex
    edges = defaultdict(dict)
    for i, vertex in enumerate(vertices):
        possible_start_path = [neighbour for neighbour in grid.four_neighbours(vertex) if grid[neighbour] != '#']
        for chosen_path in possible_start_path:
            path = [vertex, chosen_path]
            while True:
                next_steps = [neighbour for neighbour in grid.four_neighbours(path[-1]) if neighbour not in path and grid[neighbour] != '#']
                assert len(next_steps) == 1
                path.append(next_steps[0])
                if path[-1] in vertices:
                    break

            # found edge:
            vertex_2 = path[-1]
            assert vertex_2 in vertices
            id_vertex_2 = vertices.index(vertex_2)
            edges[i][id_vertex_2] = len(path) - 1
    return vertices, edges


def find_longest_path(vertices, edges):
    start_vertex = vertices[0]
    end_vertex = vertices[1]
    #dijksra
    # to find maximum we need to take negative value sof the edges
    edges_inverted = deepcopy(edges)
    for key in edges_inverted:
        for key_2 in edges_inverted[key]:
            edges_inverted[key][key_2] *= -1
    start = 0
    goal = 1
    counter = itertools.count()
    predecessors: dict[int, int | None] = {start: None}
    open = [PrioritizedState(0, next(counter), start)]
    distances_from_start: dict[int, float] = defaultdict(lambda: float("inf"))
    distances_from_start[start] = 0

    visited: set[int] = set()

    while open:
        current_state = heapq.heappop(open)
        current = current_state.vertex_idx
        visited.add(current)
        # print(f'{current=}: {_reconstruct_path(predecessors, current)}')
        if current == goal:
            path = _reconstruct_path(predecessors, current)
            path_spatial = [vertices[val] for val in path]
            # print(f'path: {path_spatial}')
            # print(f'path does not cross self: {len(path_spatial) == len(set(path_spatial))}')
            result = sum([edges[p_1][p_2] for p_1, p_2 in zip(path, path[1:])])
            # print(f'found solution: {sum([edges[p_1][p_2] for p_1, p_2 in zip(path, path[1:])])}')
            return result
        # for adjacent, distance in distance_matrix[current].items():
        for adjacent, distance in edges_inverted[current].items():

            if math.isinf(distance) or adjacent in visited:
                continue

            new_distance = distances_from_start[current] + distance
            if new_distance < distances_from_start[adjacent]:
                predecessors[adjacent] = current
                distances_from_start[adjacent] = new_distance
                heapq.heappush(open, PrioritizedState(new_distance, next(counter), adjacent))

    pass

def dfs(start, goal, grid):
    visited = {start}
    predecessors = {start: None}
    open = [start]

    while open:
        current = open.pop()

        if current == goal:
            break

        for adjacent in grid.four_neighbors(current):
            if adjacent not in visited and grid[adjacent] != "#":
                visited.add(adjacent)
                open.append(adjacent)
                predecessors[adjacent] = current

    return reconstruct_path(goal, predecessors)


FOUND_PATHS = []
MAX_VALUE = 0


def find_path_II(edges, grid, path, goal):
    global MAX_VALUE
    if path[-1] == goal:
        result = sum([edges[p_1][p_2] for p_1, p_2 in zip(path, path[1:])]) + edges[goal][1]
        if result > MAX_VALUE:
            MAX_VALUE = result
            print(f'new best path found: {path}, value: {result}, max_value: {MAX_VALUE}')
        # FOUND_PATHS.append(path)
    for adj in edges[path[-1]]:
        if adj not in path:
            find_path_II(edges, grid, path + [adj], goal)


def find_longest_path_II(vertices, edges, grid) -> None:
    start_node = 0
    end_node = 1
    edges_start_node = list(edges[start_node].keys())
    edges_end_node = list(edges[end_node].keys())
    path = [start_node, edges_start_node[0]]

    find_path_II(edges, grid, path, edges_end_node[0])
    return MAX_VALUE

    #
    # # path must start
    # for perm in tqdm(itertools.permutations(range(2, len(vertices)))):
    #     possible_edges = {(0, perm[0])} | {(p_1, p_2) for p_1, p_2 in zip(perm, perm[1:])} | {(perm[-1], 1)}
    #     if possible_edges.issubset(set_of_edges):
    #         print(f'possible path: {perm}')
    # pass


if __name__ == '__main__':
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/23.txt')
    _grid = Grid([list(line) for line in raw_data])
    _vertices, _edges = parse_graph(_grid)
    s2 = find_longest_path_II(_vertices, _edges, _grid)
    print(s2)
    # graph is
    #s1 = solve_1(_grid)
    #print(s1)
    #s2 = solve_2(_grid)
    #print(s2)
# 5954: too low
# 6322: too low but close (someone else solution)
# 6324: too low
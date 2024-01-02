import copy
import heapq
import math
from collections import defaultdict

import itertools
import numpy as np

from tqdm import tqdm

from notebooks.aoc_2023.day_16 import get_direction
from notebooks.aoc_2023.utils import load_stripped_lines, Grid, dijkstra_optimize, PrioritizedState, _reconstruct_path, COORDINATE


def build_distance_matrix_optimize(vertices: list[COORDINATE], edges: dict[COORDINATE, int]) -> list[dict[float]]:
    distances = [{i: np.inf} for i in range(len(vertices))]
    # print(distances)
    for i, v in enumerate(vertices):
        for j, u in enumerate(vertices):
            if (i, j) in edges:
                distances[i][j] = float(edges[(i, j)])

    return distances


def path_would_be_to_straight_or_crossing(predecessors, current, adjacent, max_straight, vertices):
    path_so_far = _reconstruct_path(predecessors, current)
    if adjacent in path_so_far:
        # adjacent point was already crossed
        return True
    path = [vertices[id_vertex] for id_vertex in path_so_far] + [vertices[adjacent]]
    # i am interested in the last max_straight:
    path_interested = path[-5:]
    if len(path_interested) <= 4:
        return False
    directions = [get_direction(p_2, p_1) for p_1, p_2 in zip(path_interested, path_interested[1:])]
    if directions[0] == directions[1] == directions[2] == directions[3]:
        # this is not alow
        print(f'--------------------------------')
        print(f'with path: {path}')
        print(f'{path_interested=}')
        print(f'{directions=} is not allowed')
        return True
    return False


def dijkstra_optimize_zig_zag(vertices: list, distance_matrix: list[dict[float]], start: int, goal: int, max_straight: int) -> list[int]:
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
        print(f'{current=}: {_reconstruct_path(predecessors, current)}')
        if current == goal:
            continue

        for adjacent, distance in distance_matrix[current].items():

            if math.isinf(distance) or path_would_be_to_straight_or_crossing(predecessors, current, adjacent, max_straight, vertices):
                continue

            new_distance = distances_from_start[current] + distance
            if new_distance < distances_from_start[adjacent]:
                predecessors[adjacent] = current
                distances_from_start[adjacent] = new_distance
                heapq.heappush(open, PrioritizedState(new_distance, next(counter), adjacent))

    path = _reconstruct_path(predecessors, goal)
    #for v in path:
    #    screen.addstr(vertices[v].y, vertices[v].x, ALPHABET[v], curses.color_pair(Color.MAGENTA))
    #screen.addstr(1, 0, "->".join([ALPHABET[v] for v in path]), curses.color_pair(Color.MAGENTA))
    # animate(screen)
    #screen.nodelay(False)
    #screen.getkey()
    return path


def visualize_path(_p: list[int], grid_height, grid_width):
    _vis = Grid([['.' for _ in range(grid_width)] for _ in range(grid_height)])
    for val in _p:
        _vis[vertices[val]] = '#'
    return _vis


def visualize_path_with_number(_p: list[int], grid):
    _vis = copy.deepcopy(grid)
    for val in _p:
        _vis[vertices[val]] = '#'
    return _vis


if __name__ == "__main__":
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/17_test.txt')
    _grid = Grid([[int(item) for item in list(line)] for line in raw_data])
    start_point = (0, 0)
    end_point = (_grid.height - 1, _grid.width - 1)
    # value on start is 0:
    _grid[start_point] = 0

    vertices = [(j, i) for i in range(_grid.width) for j in range(_grid.height)]
    edges: dict[tuple[int, int], float] = {}
    for vertex in tqdm(vertices):
        vertex_id = vertices.index(vertex)
        # if vertex_id == 0:
        #     edges[(vertex_id, 13)] = 4
        #     continue
        for neighbour in _grid.four_neighbours(vertex):
            neighbour_id = vertices.index(neighbour)
            edges[(vertex_id, neighbour_id)] = _grid[neighbour]

    pass
    distance_matrix = build_distance_matrix_optimize(vertices, edges)
    start = vertices.index(start_point)
    goal = vertices.index(end_point)
    found_path = dijkstra_optimize_zig_zag(vertices, distance_matrix, start=start, goal=goal, max_straight=3)

    _vis_found_path = visualize_path(found_path, _grid.height, _grid.width)

    heat_losses = [_grid[vertices[val]] for val in found_path]

    correct_path = [0, 13, 26, 27, 40, 53, 66, 65, 78, 91, 104, 105,  106, 119, 132, 133, 134, 147, 148, 149, 150, 163, 164, 165, 166, 153, 154, 155, 168]
    _vis_correct = visualize_path_with_number(correct_path, _grid)
    correct_heat_loss = [_grid[vertices[val]] for val in correct_path]

    pass
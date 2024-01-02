import copy
import heapq
import math
from collections import defaultdict

import itertools
import numpy as np

from tqdm import tqdm

from notebooks.aoc_2023.day_16 import get_direction
from notebooks.aoc_2023.utils import load_stripped_lines, Grid, dijkstra_optimize, PrioritizedState, _reconstruct_path, COORDINATE

import sys
sys.setrecursionlimit(100000)


def build_distance_matrix_optimize(vertices: list[COORDINATE], edges: dict[COORDINATE, int]) -> list[dict[float]]:
    distances = [{i: np.inf} for i in range(len(vertices))]
    # print(distances)
    for i, v in enumerate(vertices):
        for j, u in enumerate(vertices):
            if (i, j) in edges:
                distances[i][j] = float(edges[(i, j)])

    return distances


def path_would_be_to_straight_or_crossing(path_so_far: list[COORDINATE], next_point: COORDINATE):
    if next_point in path_so_far:
        return True
    new_path = path_so_far + [next_point]
    # i am interested in the last max_straight:
    path_interested = new_path[-5:]
    if len(path_interested) <= 4:
        return False
    directions = [get_direction(p_2, p_1) for p_1, p_2 in zip(path_interested, path_interested[1:])]
    if directions[0] == directions[1] == directions[2] == directions[3]:
        # # this is not alow
        # print(f'--------------------------------')
        # print(f'with path: {new_path}')
        # print(f'{path_interested=}')
        # print(f'{directions=} is not allowed')
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
            pass
            break

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


def visualize_path(_p: list[COORDINATE], grid_height, grid_width):
    _vis = Grid([['.' for _ in range(grid_width)] for _ in range(grid_height)])
    for i, val in enumerate(_p):
        _vis[val] = i
    return _vis


def visualize_path_with_number(_p: list[int], grid):
    _vis = copy.deepcopy(grid)
    for val in _p:
        _vis[vertices[val]] = '#'
    return _vis


BEST_PATH = []
BEST_PATH_VALUE = 1300


def manhattan_distance(p_1, p_2) -> int:
    return abs(p_1[0] - p_2[0]) + abs(p_1[1] - p_2[1])


def priorite_neighbours(current, end, grid):
    four_points = grid.four_neighbours(current)
    # fort the four points to be as close to
    return sorted(four_points, key=lambda x: abs((x[0] - end[0])**2 + (x[1] - end[0])**2))


def find_solution_using_bf(grid: Grid, end: COORDINATE, trajectory: list[COORDINATE]):
    global BEST_PATH_VALUE
    # print(trajectory)
    total_heat_loss = sum([_grid[val] for val in trajectory])
    current_point = trajectory[-1]
    if current_point == end:
        # print(f'solution found: {trajectory}')
        # solution found:
        if total_heat_loss < BEST_PATH_VALUE:
            BEST_PATH_VALUE = total_heat_loss
            BEST_PATH = trajectory
            _vis = visualize_path(BEST_PATH, _grid.height, _grid.width)
            print(f'new solution found: {total_heat_loss}, traj: {BEST_PATH}')
        return True

    if BEST_PATH_VALUE is not None and total_heat_loss > BEST_PATH_VALUE:
        # no point of continuing
        # print(f'total_heat_loss not good enough {total_heat_loss}')
        return False

    if BEST_PATH_VALUE is not None and (total_heat_loss + manhattan_distance(trajectory[-1], end) > BEST_PATH_VALUE):
        # print(f'there is no way er can get to distance with better score, no point of continuing')
        return False

    if BEST_PATH_VALUE is not None and (total_heat_loss + manhattan_distance(trajectory[-1], end) > BEST_PATH_VALUE):
        # print(f'there is no way er can get to distance with better score, no point of continuing')
        return False
    # if BEST_PATH_VALUE is not None and (total_heat_loss + shortest_path(trajectory[-1], ) > BEST_PATH_VALUE):
    #     # print(f'there is no way er can get to distance with better score, no point of continuing')
    #     return False

    for neighbour in priorite_neighbours(current_point, end_point, grid):
        if path_would_be_to_straight_or_crossing(trajectory, neighbour):
            continue
        find_solution_using_bf(grid, end, trajectory + [neighbour])


if __name__ == "__main__":
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/17.txt')
    _grid = Grid([[int(item) for item in list(line)] for line in raw_data])
    start_point = (0, 0)
    end_point = (_grid.height - 1, _grid.width - 1)
    # value on start is 0:
    _grid[start_point] = 0
    _trajectorory = [start_point]
    a = find_solution_using_bf(_grid, end_point, _trajectorory)
    vertices = [(j, i) for i in range(_grid.width) for j in range(_grid.height)]
    correct_path = [0, 13, 26, 27, 40, 53, 66, 65, 78, 91, 104, 105,  106, 119, 132, 133, 134, 147, 148, 149, 150, 163, 164, 165, 166, 153, 154, 155, 168]
    _vis_correct = visualize_path_with_number(correct_path, _grid)
    correct_heat_loss = [_grid[vertices[val]] for val in correct_path]

# geuess: 1414 - wrong
# 1300: to high
# new solution found: 1409, traj: [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6), (7, 7), (8, 7), (8, 8), (9, 8), (9, 9), (10, 9), (10, 10), (11, 10), (11, 11), (12, 11), (12, 12), (13, 12), (13, 13), (14, 13), (14, 14), (15, 14), (15, 15), (16, 15), (16, 16), (17, 16), (17, 17), (18, 17), (18, 18), (19, 18), (19, 19), (20, 19), (20, 20), (21, 20), (21, 21), (22, 21), (22, 22), (23, 22), (23, 23), (24, 23), (24, 24), (25, 24), (25, 25), (26, 25), (26, 26), (27, 26), (27, 27), (28, 27), (28, 28), (29, 28), (29, 29), (30, 29), (30, 30), (31, 30), (31, 31), (32, 31), (32, 32), (33, 32), (33, 33), (34, 33), (34, 34), (35, 34), (35, 35), (36, 35), (36, 36), (37, 36), (37, 37), (38, 37), (38, 38), (39, 38), (39, 39), (40, 39), (40, 40), (41, 40), (41, 41), (42, 41), (42, 42), (43, 42), (43, 43), (44, 43), (44, 44), (45, 44), (45, 45), (46, 45), (46, 46), (47, 46), (47, 47), (48, 47), (48, 48), (49, 48), (49, 49), (50, 49), (50, 50), (51, 50), (51, 51), (52, 51), (52, 52), (53, 52), (53, 53), (54, 53), (54, 54), (55, 54), (55, 55), (56, 55), (56, 56), (57, 56), (57, 57), (58, 57), (58, 58), (59, 58), (59, 59), (60, 59), (60, 60), (61, 60), (61, 61), (62, 61), (62, 62), (63, 62), (63, 63), (64, 63), (64, 64), (65, 64), (65, 65), (66, 65), (66, 66), (67, 66), (67, 67), (68, 67), (68, 68), (69, 68), (69, 69), (70, 69), (70, 70), (71, 70), (71, 71), (72, 71), (72, 72), (73, 72), (73, 73), (74, 73), (74, 74), (75, 74), (75, 75), (76, 75), (76, 76), (77, 76), (77, 77), (78, 77), (78, 78), (79, 78), (79, 79), (80, 79), (80, 80), (81, 80), (81, 81), (82, 81), (82, 82), (83, 82), (83, 83), (84, 83), (84, 84), (85, 84), (85, 85), (86, 85), (86, 86), (87, 86), (87, 87), (88, 87), (88, 88), (89, 88), (89, 89), (90, 89), (90, 90), (91, 90), (91, 91), (92, 91), (92, 92), (93, 92), (93, 93), (94, 93), (94, 94), (95, 94), (95, 95), (96, 95), (96, 96), (97, 96), (97, 97), (98, 97), (98, 98), (99, 98), (99, 99), (100, 99), (100, 100), (101, 100), (101, 101), (102, 101), (102, 102), (103, 102), (103, 103), (104, 103), (104, 104), (105, 104), (105, 105), (106, 105), (106, 106), (107, 106), (107, 107), (108, 107), (108, 108), (109, 108), (109, 109), (110, 109), (110, 110), (111, 110), (111, 111), (112, 111), (112, 112), (113, 112), (113, 113), (114, 113), (114, 114), (115, 114), (115, 115), (116, 115), (116, 116), (117, 116), (117, 117), (118, 117), (118, 118), (118, 119), (119, 119), (119, 120), (119, 121), (119, 122), (120, 122), (121, 122), (122, 122), (122, 123), (122, 124), (123, 124), (123, 125), (123, 126), (124, 126), (125, 126), (125, 127), (125, 128), (126, 128), (127, 128), (128, 128), (128, 129), (129, 129), (129, 130), (129, 131), (129, 132), (128, 132), (128, 133), (128, 134), (129, 134), (129, 135), (130, 135), (130, 136), (130, 137), (131, 137), (132, 137), (133, 137), (133, 138), (134, 138), (135, 138), (136, 138), (136, 139), (137, 139), (138, 139), (139, 139), (139, 140), (140, 140)]
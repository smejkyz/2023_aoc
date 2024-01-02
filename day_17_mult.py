import copy
import heapq
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import math
from collections import defaultdict

import itertools
import numpy as np

from tqdm import tqdm

from notebooks.aoc_2023.utils import load_stripped_lines, Grid, dijkstra_optimize, PrioritizedState, _reconstruct_path, COORDINATE
'''
You need to apply Dijkstra algorithm to a different graph. 
Your “position” in the new graph is not just where you are, 
but also what direction you’re going in and how long you’ve been going in that direction.
 Then you can figure out the new legal moves just given your current position/state
  (namely: if you’re already been going on the current direction for 3 moves, you can’t continue).

The new graph is 12 times larger than the original graph, 
because for every grid square you could have gotten there in any of 4 directions 
and have been traveling in that direction for 1 or 2 or 3 moves.

Then you have 12 possible destinations - anything where you’re at the bottom-right grid square. But that’s fine - your final answer is just the minimum distance to any of those 12.
'''


class Direction(Enum):
    RIGHT = 'right'
    LEFT = 'left'
    UP = 'up'
    DOWN = 'down'
    NONE = 'none'  # starting point


MAP_TUPLE_DIR_TO_DIRECTION: dict[COORDINATE, Direction] = {
    (0, -1): Direction.LEFT,
    (0, 1): Direction.RIGHT,
    (1, 0): Direction.DOWN,
    (-1, 0): Direction.UP
}

MAP_DO_NOT_GO_BACK: dict[Direction, Direction] = {
    Direction.RIGHT: Direction.LEFT,
    Direction.LEFT: Direction.RIGHT,
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.NONE: Direction.NONE,
}


def get_direction(current_point: COORDINATE, neighbour: COORDINATE) -> Direction:
    tuple_direction = neighbour[0] - current_point[0], neighbour[1] - current_point[1]
    return MAP_TUPLE_DIR_TO_DIRECTION[tuple_direction]


@dataclass
class Vertex:
    coordinate: COORDINATE
    direction: Direction  # from which direction did I get here?
    nb_same_direction: int  # 1, 2,3


def build_distance_matrix_optimize(vertices: list[COORDINATE], edges: dict[COORDINATE, int]) -> list[dict[float]]:
    distances = [{i: np.inf} for i in range(len(vertices))]
    # print(distances)
    for i, j in edges:
        distances[i][j] = edges[(i, j)]
    # for i, v in enumerate(vertices):
    #     for j, u in enumerate(vertices):
    #         if (i, j) in edges:
    #             distances[i][j] = float(edges[(i, j)])

    return distances


def compute_adjacent(_vertices: list[Vertex], grid: Grid, current: int) -> list[tuple[int, float]]:
    current_vertex = _vertices[current]
    neighbours_spatial_coordinates = grid.four_neighbours(current_vertex.coordinate)
    adjacent = []
    for neighbour_spatial_coor in neighbours_spatial_coordinates:
        neighbour_vertex = None
        neighbour_direction = get_direction(current_vertex.coordinate, neighbour_spatial_coor)
        if neighbour_direction == current_vertex.direction:
            if current_vertex.nb_same_direction < 3:
                # is possible to continue in the same direction - no edge added
                neighbour_vertex = Vertex(neighbour_spatial_coor, neighbour_direction, current_vertex.nb_same_direction + 1)
        elif neighbour_direction == MAP_DO_NOT_GO_BACK[current_vertex.direction]:
            # I would go back - i do not want this
            continue
        else:
            # new direction:
            neighbour_vertex = Vertex(neighbour_spatial_coor, neighbour_direction, 1)
        if neighbour_vertex is not None and neighbour_spatial_coor != (0, 0):  # do not return back to start
            neighbour_id = _vertices.index(neighbour_vertex)
            adjacent.append((neighbour_id, grid[neighbour_spatial_coor]))
    return adjacent


def move_in_direction(current: COORDINATE, direction: Direction) -> COORDINATE:
    if direction == Direction.RIGHT:
        return current[0], current[1] + 1
    if direction == Direction.LEFT:
        return current[0], current[1] - 1
    if direction == Direction.DOWN:
        return current[0] + 1, current[1]
    if direction == Direction.UP:
        return current[0] - 1, current[1]
    raise NotImplemented()


def compute_adjacent_part_two(_vertices: list[Vertex], grid: Grid, current: int) -> list[tuple[int, float]]:
    current_vertex = _vertices[current]

    if 1 <= current_vertex.nb_same_direction < 4:
        # it can only move in the same direction:
        neighbour_spatial_coor = move_in_direction(current_vertex.coordinate, current_vertex.direction)
        if not grid.is_inside_grid(neighbour_spatial_coor) or neighbour_spatial_coor == (0, 0):
            # not possible to continue:
            return []
        neighbour_vertex = Vertex(neighbour_spatial_coor, current_vertex.direction, current_vertex.nb_same_direction + 1)
        neighbour_id = _vertices.index(neighbour_vertex)
        return [(neighbour_id, grid[neighbour_spatial_coor])]
    # otherwise it is the same - 5 - 9 can move in any direction, 10 has to turn
    neighbours_spatial_coordinates = grid.four_neighbours(current_vertex.coordinate)
    adjacent = []
    for neighbour_spatial_coor in neighbours_spatial_coordinates:
        neighbour_vertex = None
        neighbour_direction = get_direction(current_vertex.coordinate, neighbour_spatial_coor)
        if neighbour_direction == current_vertex.direction:
            if current_vertex.nb_same_direction < 10:
                # is possible to continue in the same direction - no edge added
                neighbour_vertex = Vertex(neighbour_spatial_coor, neighbour_direction, current_vertex.nb_same_direction + 1)
        elif neighbour_direction == MAP_DO_NOT_GO_BACK[current_vertex.direction]:
            # I would go back - i do not want this
            continue
        else:
            # new direction:
            neighbour_vertex = Vertex(neighbour_spatial_coor, neighbour_direction, 1)
        if neighbour_vertex is not None and neighbour_spatial_coor != (0, 0):  # do not return back to start
            neighbour_id = _vertices.index(neighbour_vertex)
            adjacent.append((neighbour_id, grid[neighbour_spatial_coor]))
    return adjacent


def dijkstra_optimize_zig_zag(_vertices: list, grid, start: int, goals: list[int]) -> list[int]:
    counter = itertools.count()

    predecessors: dict[int, int | None] = {start: None}
    open = [PrioritizedState(0, next(counter), start)]
    distances_from_start: dict[int, float] = defaultdict(lambda: float("inf"))
    distances_from_start[start] = 0

    visited: set[int] = set()

    while open:
        if len(visited) % 1000 == 0:
            print(len(visited) / len(_vertices) * 100)
        current_state = heapq.heappop(open)
        current = current_state.vertex_idx
        visited.add(current)
        # print(f'{current=}: {_reconstruct_path(predecessors, current)}')
        if current in goals:
            path = _reconstruct_path(predecessors, current)
            path_spatial = [_vertices[val].coordinate for val in path]
            print(f'path: {path_spatial}')
            print(f'path does not cross self: {len(path_spatial) == len(set(path_spatial))}')
            print(f'found solution: {sum([_grid[_vertices[val].coordinate] for val in path])}')

        # for adjacent, distance in distance_matrix[current].items():
        for adjacent, distance in compute_adjacent(_vertices, grid, current):

            if math.isinf(distance):
                continue

            new_distance = distances_from_start[current] + distance
            if new_distance < distances_from_start[adjacent]:
                predecessors[adjacent] = current
                distances_from_start[adjacent] = new_distance
                heapq.heappush(open, PrioritizedState(new_distance, next(counter), adjacent))
    print(f'visited: {len(visited)} which is {len(visited) / len(_vertices) * 100} percent')
    paths = []
    for single_goal in goals:
        try:
            paths.append(_reconstruct_path(predecessors, single_goal))
        except:
            print(f'path not foud for goal: {single_goal}')
    #for v in path:
    #    screen.addstr(vertices[v].y, vertices[v].x, ALPHABET[v], curses.color_pair(Color.MAGENTA))
    #screen.addstr(1, 0, "->".join([ALPHABET[v] for v in path]), curses.color_pair(Color.MAGENTA))
    # animate(screen)
    #screen.nodelay(False)
    #screen.getkey()
    return paths


def heuristic(a: Vertex, b: Vertex) -> int:
    return abs(a.coordinate[1] - b.coordinate[1]) + abs(a.coordinate[0] - b.coordinate[0])


def a_star_zig_zag(_vertices: list[Vertex], grid: Grid, start: int, goals: list[int]) -> list[list[int]]:
    _now = datetime.now()
    print(f'starting A* {_now}')
    counter = itertools.count()

    predecessors: dict[int, int | None] = {start: None}
    open = [PrioritizedState(0, next(counter), start)]
    distances_from_start: dict[int, float] = defaultdict(lambda: float("inf"))
    distances_from_start[start] = 0

    visited: set[int] = set()

    while open:
        if len(visited) % 1000 == 0:
            print(f'{len(visited) / len(_vertices) * 100} percent visited, elapsed time: {datetime.now() - _now}')
        current_state = heapq.heappop(open)
        current = current_state.vertex_idx
        visited.add(current)
        if current in goals:
            path = _reconstruct_path(predecessors, current)
            path_spatial = [_vertices[val].coordinate for val in path]
            print(f'path: {path_spatial}')
            print(f'path does not cross self: {len(path_spatial) == len(set(path_spatial))}')
            print(f'found solution: {sum([grid[_vertices[val].coordinate] for val in path])}')
            _vis_path = visualize_path(path_spatial, grid.height, grid.width)
            # for p in path:
            #     print(_vertices[p])

        # for adjacent, distance in distance_matrix[current].items():
        # for adjacent, distance in compute_adjacent(_vertices, grid, current): # for part one
        for adjacent, distance in compute_adjacent_part_two(_vertices, grid, current):

            if math.isinf(distance):
                continue

            g_score = distances_from_start[current] + distance
            if g_score < distances_from_start[adjacent]:
                predecessors[adjacent] = current
                distances_from_start[adjacent] = g_score
                f_score = g_score + heuristic(_vertices[adjacent], _vertices[goals[0]])
                heapq.heappush(open, PrioritizedState(f_score, next(counter), adjacent))
    print(f'visited: {len(visited)} which is {len(visited) / len(_vertices) * 100} percent')
    paths = []
    for single_goal in goals:
        try:
            paths.append(_reconstruct_path(predecessors, single_goal))
        except:
            print(f'path not foud for goal: {single_goal}')
    #for v in path:
    #    screen.addstr(vertices[v].y, vertices[v].x, ALPHABET[v], curses.color_pair(Color.MAGENTA))
    #screen.addstr(1, 0, "->".join([ALPHABET[v] for v in path]), curses.color_pair(Color.MAGENTA))
    # animate(screen)
    #screen.nodelay(False)
    #screen.getkey()
    return paths


def visualize_path(_p: list[COORDINATE], grid_height, grid_width):
    _vis = Grid([['.' for _ in range(grid_width)] for _ in range(grid_height)])
    for val in _p:
        _vis[val] = '#'
    return _vis


def visualize_path_with_number(_p: list[int], grid):
    _vis = copy.deepcopy(grid)
    for val in _p:
        _vis[vertices[val]] = '#'
    return _vis


def compute_edges(_vertices: list[Vertex], grid: Grid) -> dict[tuple[int, int], float]:
    _edges: dict[tuple[int, int], float] = {}
    for vertex in tqdm(_vertices):
        vertex_id = _vertices.index(vertex)
        neighbours_spatial_coordinates = _grid.four_neighbours(vertex.coordinate)
        for neighbour_spatial_coor in neighbours_spatial_coordinates:
            neighbour_vertex = None
            neighbour_direction = get_direction(vertex.coordinate, neighbour_spatial_coor)
            if neighbour_direction == vertex.direction:
                if vertex.nb_same_direction < 3:
                    # is possible to continue in the same direction - no edge added
                    neighbour_vertex = Vertex(neighbour_spatial_coor, neighbour_direction, vertex.nb_same_direction + 1)
            elif neighbour_direction == MAP_DO_NOT_GO_BACK[vertex.direction]:
                # I would go back - i do not want this
                continue
            else:
                # new direction:
                neighbour_vertex = Vertex(neighbour_spatial_coor, neighbour_direction, 1)
            if neighbour_vertex is not None:
                neighbour_id = _vertices.index(neighbour_vertex)
                _edges[(vertex_id, neighbour_id)] = grid[neighbour_spatial_coor]
    return _edges


def solve_1(grid: Grid) -> int:
    start_point = (0, 0)
    end_point = (grid.height - 1, grid.width - 1)
    # value on start is 0:
    grid[start_point] = 0

    # create Graph:
    vertices = []
    for y, x in itertools.product(range(grid.height), range(grid.width)):
        if y == 0 and x == 0:
            vertices.append(Vertex((y, x), Direction.NONE, 0))
            continue
        for direction in list(Direction):
            for nb_same_direction in range(1, 4):
                vertices.append(Vertex((y, x), direction, nb_same_direction))

    start_vertex = Vertex(start_point, Direction.NONE, 0)
    start = vertices.index(start_vertex)
    goals = [i for i, vert in enumerate(vertices) if vert.coordinate == end_point]
    found_paths = a_star_zig_zag(vertices, grid, start=start, goals=goals)
    heat_losses = [sum([grid[vertices[val].coordinate] for val in path]) for path in found_paths]
    print(min(heat_losses))


def solve_2(grid: Grid) -> int:
    start_point = (0, 0)
    end_point = (grid.height - 1, grid.width - 1)
    # value on start is 0:
    grid[start_point] = 0

    # create Graph:
    vertices = []
    for y, x in itertools.product(range(grid.height), range(grid.width)):
        if y == 0 and x == 0:
            vertices.append(Vertex((y, x), Direction.NONE, 0))
            continue
        for direction in list(Direction):
            for nb_same_direction in range(1, 11):
                vertices.append(Vertex((y, x), direction, nb_same_direction))

    start_vertex = Vertex(start_point, Direction.NONE, 0)
    start = vertices.index(start_vertex)
    goals = [i for i, vert in enumerate(vertices) if vert.coordinate == end_point and vert.nb_same_direction >= 4]
    found_paths = a_star_zig_zag(vertices, grid, start=start, goals=goals)
    heat_losses = [sum([grid[vertices[val].coordinate] for val in path]) for path in found_paths]
    print(min(heat_losses))


if __name__ == "__main__":
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/17.txt')
    main_grid = Grid([[int(item) for item in list(line)] for line in raw_data])
    # solve_1(main_grid)
    solve_2(main_grid)

    heat_losses = []
    #

    #found_path = [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2), (0, 3), (0, 4), (1, 4), (1, 5), (1, 6), (2, 6), (2, 7), (2, 8), (3, 8), (3, 9), (2, 9), (2, 10), (2, 11), (2, 12), (1, 12), (1, 13), (1, 14), (2, 14), (2, 15), (2, 16), (1, 16), (1, 17), (1, 18), (1, 19), (2, 19), (2, 20), (2, 21), (3, 21), (3, 22), (3, 23), (3, 24), (4, 24), (4, 25), (4, 26), (4, 27), (5, 27), (5, 28), (5, 29), (5, 30), (6, 30), (6, 31), (7, 31), (7, 32), (7, 33), (7, 34), (6, 34), (6, 35), (6, 36), (6, 37), (5, 37), (5, 38), (6, 38), (6, 39), (6, 40), (6, 41), (7, 41), (8, 41), (8, 42), (8, 43), (8, 44), (9, 44), (9, 45), (10, 45), (10, 46), (10, 47), (10, 48), (11, 48), (11, 49), (12, 49), (12, 50), (12, 51), (11, 51), (11, 52), (11, 53), (11, 54), (10, 54), (10, 55), (9, 55), (9, 56), (8, 56), (8, 57), (8, 58), (7, 58), (7, 59), (7, 60), (7, 61), (6, 61), (6, 62), (7, 62), (8, 62), (9, 62), (9, 63), (9, 64), (9, 65), (10, 65), (10, 66), (10, 67), (11, 67), (12, 67), (12, 68), (12, 69), (13, 69), (13, 70), (13, 71), (13, 72), (12, 72), (12, 73), (12, 74), (12, 75), (11, 75), (11, 76), (11, 77), (11, 78), (10, 78), (9, 78), (9, 79), (9, 80), (9, 81), (8, 81), (7, 81), (7, 82), (7, 83), (7, 84), (8, 84), (8, 85), (9, 85), (9, 86), (9, 87), (8, 87), (8, 88), (8, 89), (8, 90), (9, 90), (9, 91), (10, 91), (10, 92), (10, 93), (10, 94), (11, 94), (12, 94), (13, 94), (13, 95), (13, 96), (13, 97), (14, 97), (15, 97), (16, 97), (16, 98), (17, 98), (17, 99), (18, 99), (19, 99), (19, 100), (19, 101), (19, 102), (20, 102), (21, 102), (21, 103), (21, 104), (21, 105), (22, 105), (23, 105), (24, 105), (24, 106), (24, 107), (24, 108), (25, 108), (26, 108), (27, 108), (27, 109), (28, 109), (28, 110), (28, 111), (29, 111), (30, 111), (30, 112), (30, 113), (30, 114), (31, 114), (31, 115), (31, 116), (32, 116), (32, 117), (32, 118), (33, 118), (33, 119), (34, 119), (35, 119), (35, 120), (36, 120), (37, 120), (37, 121), (37, 122), (37, 123), (38, 123), (38, 124), (39, 124), (39, 125), (39, 126), (39, 127), (40, 127), (40, 128), (40, 129), (40, 130), (41, 130), (42, 130), (42, 131), (43, 131), (43, 132), (44, 132), (44, 133), (45, 133), (46, 133), (47, 133), (47, 134), (47, 135), (48, 135), (49, 135), (49, 136), (50, 136), (51, 136), (52, 136), (52, 137), (53, 137), (54, 137), (55, 137), (55, 138), (56, 138), (57, 138), (57, 137), (58, 137), (59, 137), (60, 137), (60, 136), (61, 136), (62, 136), (63, 136), (63, 135), (63, 134), (64, 134), (65, 134), (66, 134), (66, 135), (67, 135), (68, 135), (68, 136), (69, 136), (70, 136), (71, 136), (71, 137), (72, 137), (72, 138), (73, 138), (74, 138), (75, 138), (75, 137), (76, 137), (77, 137), (78, 137), (78, 138), (79, 138), (80, 138), (81, 138), (81, 139), (82, 139), (83, 139), (84, 139), (84, 138), (85, 138), (86, 138), (86, 137), (87, 137), (88, 137), (89, 137), (89, 136), (90, 136), (91, 136), (91, 137), (92, 137), (92, 138), (93, 138), (94, 138), (94, 139), (95, 139), (96, 139), (97, 139), (97, 140), (98, 140), (99, 140), (100, 140), (100, 139), (101, 139), (102, 139), (103, 139), (103, 138), (104, 138), (104, 137), (105, 137), (105, 136), (105, 135), (106, 135), (106, 134), (107, 134), (108, 134), (109, 134), (109, 133), (109, 132), (110, 132), (111, 132), (112, 132), (112, 133), (113, 133), (114, 133), (114, 134), (115, 134), (116, 134), (117, 134), (117, 135), (118, 135), (119, 135), (120, 135), (120, 136), (121, 136), (122, 136), (123, 136), (123, 137), (124, 137), (125, 137), (125, 136), (126, 136), (126, 135), (127, 135), (128, 135), (129, 135), (129, 136), (130, 136), (130, 137), (131, 137), (132, 137), (133, 137), (133, 138), (134, 138), (135, 138), (136, 138), (136, 139), (137, 139), (138, 139), (139, 139), (139, 140), (140, 140)]
    #heat_loss = sum([main_grid[val] for val in found_path[1:]])
    #_vis_found = visualize_path(found_path, main_grid.height, main_grid.width)
    # correct_heat_loss = [_grid[vertices[val]] for val in correct_path]

    pass

# 817: not correct
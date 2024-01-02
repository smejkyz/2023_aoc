import numpy as np

from notebooks.aoc_2023.day_17_mult import Direction
from notebooks.aoc_2023.utils import load_stripped_lines, Grid, COORDINATE


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


if __name__ == '__main__':
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/23.txt')
    _grid = Grid([list(line) for line in raw_data])
    #s1 = solve_1(_grid)
    #print(s1)
    s2 = solve_2(_grid)
    print(s2)
# 5954: too low
# 6322: too low but close (someone else solution)
# 6324: too low
from collections import defaultdict
from typing import Iterable

import itertools
from copy import copy

import numpy as np

from notebooks.aoc_2023.utils import load_stripped_lines, Grid, COORDINATE

FINAL_POINTS = []


def solve_grid(grid, visited_path: list[COORDINATE], final_nb_steps: int) -> bool:
    if len(visited_path) - 1 == final_nb_steps:
        FINAL_POINTS.append(visited_path[-1])
        return True
    current_position = visited_path[-1]
    for neighbour in grid.four_neighbours(current_position):
        if grid[neighbour] == '#':
            continue
        solve_grid(grid, visited_path + [neighbour], final_nb_steps)


def solve_1(grid: Grid, steps: int) -> int:
    starting_position = grid.get_coordinates('S')[0]
    solve_grid(grid, [starting_position], final_nb_steps=steps)
    _visited_grid = copy(grid)
    return len(set(FINAL_POINTS))
    # for coor in FINAL_POINTS:
    #     _visited_grid[coor] = 'O'
    # pass


class System:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.start_symbol = 'S'
        self.rock_symbol = '#'
        self.plot_symbol = '.'
        self.current_position = 'O'

    def transform(self, times: int) -> None:
        for _ in range(times):
            self._transform_single_time()

    def _transform_single_time(self) -> None:
        current_coordinates = self.grid.get_coordinates(self.current_position)
        if not current_coordinates:
            # first step:
            current_coordinates = self.grid.get_coordinates(self.start_symbol)
        # 1. remove all current coordinas from the grid:
        self.remove_coordinates(current_coordinates)

        # expand each coordinate
        for coor in current_coordinates:
            self.expand_coordinate(coor)

    def remove_coordinates(self, coordinates: list[COORDINATE]) -> None:
        for coor in coordinates:
            self.grid[coor] = self.plot_symbol

    def expand_coordinate(self, coordinate: COORDINATE) -> None:
        for neighbour in self.grid.four_neighbours(coordinate):
            if self.grid[neighbour] == self.rock_symbol:
                continue
            else:
                self.grid[neighbour] = self.current_position

    def get_positions(self) -> int:
        return len(self.grid.get_coordinates(self.current_position))


def populate_infinite(_input: COORDINATE) -> list[COORDINATE]:
    return _grid.infinite_four_neighbour(_input)


def solve_2(grid: Grid):

    current_positions = set(grid.get_coordinates('S'))
    interesting_steps = [65, 65 + 131, 65 + 2 * 131, 65 + 3 * 131, 65 + 4 * 131, 65 + 5 * 131, 65 + 6 * 131, 65 + 7 * 131]
    for steps in range(1, max(interesting_steps)+1):
        current_positions = set(itertools.chain(*[populate_infinite(pos) for pos in current_positions]))
        if steps in interesting_steps:
            print(f'steps {steps}: {len(current_positions)}')
    return len(current_positions)


def solve_2_np(grid: Grid):
    _v_func = np.vectorize(populate, otypes=[tuple])
    current_positions = np.array(grid.get_coordinates('S')).flatten()
    _testing = _v_func(current_positions[::2], current_positions[1::2])[0].flatten()
    for _ in range(64):
        # current_positions_y, current_positions_x = _v_func(current_positions[0], current_positions[1])
        #
        #
        _testing = np.array([x for x in _v_func(_testing[::2], _testing[1::2])]).flatten()
        pass


if __name__ == '__main__':
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/21.txt')
    _grid = Grid([[item for item in list(line)] for line in raw_data])
    # system = System(_grid)
    # system.transform(64)
    # s1 = system.get_positions()
    # print(s1)

    x = [0, 1, 2, 3]
    y = [3821, 34234, 94963, 186008]
    result = np.polyfit(x, y, 2)
    print(solve_2(_grid))

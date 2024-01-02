import copy
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

from notebooks.aoc_2023.utils import load_raw_lines, load_stripped_lines, Grid


def till_one_colum_north(_platform: Grid, id_col: int) -> None:
    for id_row, value in enumerate(_platform.get_column(id_col)):
        if value in ('#', '.'):
            continue
        assert value == 'O'
        roll_to = id_row
        while roll_to - 1 >= 0 and _platform[(roll_to - 1, id_col)] == '.':
            roll_to = roll_to - 1
        if roll_to != id_row:
            _platform[(roll_to, id_col)] = 'O'
            _platform[(id_row, id_col)] = '.'


def till_one_row_west(_platform: Grid, id_row: int) -> None:
    for id_col, value in enumerate(_platform.get_row(id_row)):
        if value in ('#', '.'):
            continue
        assert value == 'O'
        roll_to = id_col
        while roll_to - 1 >= 0 and _platform[(id_row, roll_to - 1)] == '.':
            roll_to = roll_to - 1
        if roll_to != id_col:
            _platform[(id_row, roll_to)] = 'O'
            _platform[(id_row, id_col)] = '.'


def till_one_row_east(_platform: Grid, id_row: int) -> None:
    for id_col, value in reversed(list(enumerate(_platform.get_row(id_row)))):
        if value in ('#', '.'):
            continue
        assert value == 'O'
        roll_to = id_col
        while roll_to + 1 < _platform.width and _platform[(id_row, roll_to + 1)] == '.':
            roll_to = roll_to + 1
        if roll_to != id_col:
            _platform[(id_row, roll_to)] = 'O'
            _platform[(id_row, id_col)] = '.'


def till_one_colum_south(_platform: Grid, id_col: int) -> None:
    for id_row, value in reversed(list(enumerate(_platform.get_column(id_col)))):
        if value in ('#', '.'):
            continue
        assert value == 'O'
        roll_to = id_row
        while roll_to + 1 < _platform.height and _platform[(roll_to + 1, id_col)] == '.':
            roll_to = roll_to + 1
        if roll_to != id_row:
            _platform[(roll_to, id_col)] = 'O'
            _platform[(id_row, id_col)] = '.'


def till_north(_platform):
    for id_col in range(_platform.width):
        till_one_colum_north(_platform, id_col)


def till_west(_platform):
    for id_row in range(_platform.height):
        till_one_row_west(_platform, id_row)


def till_south(_platform):
    for id_col in range(_platform.width):
        till_one_colum_south(_platform, id_col)


def till_east(_platform):
    for id_row in range(_platform.height):
        till_one_row_east(_platform, id_row)


def compute_total_load(_platform: Grid) -> int:
    height = _platform.height
    position = _platform.get_coordinates('O')
    individual_loads = [height-pos[0] for pos in position]
    return sum(individual_loads)


def solve_1(_platform: Grid) -> int:
    _counter_begin, _begin_values = np.unique(_platform.as_numpy, return_counts=True)
    for id_col in range(_platform.width):
        till_one_colum_north(_platform, id_col)
        assert np.all(_begin_values == np.unique(_platform.as_numpy, return_counts=True)[1])
    return compute_total_load(_platform)


def one_cycle(_platform):
    till_north(_platform)
    till_west(_platform)
    till_south(_platform)
    till_east(_platform)


def solve_2(_platform: Grid) -> int:
    _counter_begin, _begin_values = np.unique(_platform.as_numpy, return_counts=True)
    map_of_positions = defaultdict(list)
    total_iterations = 1000000000
    for iteration in tqdm(range(total_iterations)):
        # print(f'iter: {iteration}: load: {compute_total_load(_platform)}')
        key = frozenset(_platform.get_coordinates('O'))
        map_of_positions[key].append((iteration, copy.deepcopy(_platform)))
        if True and any(len(val) == 2 for val in map_of_positions.values()):
            # period found:
            _val = [(val[0][0], val[1][0]) for val in map_of_positions.values() if len(val) == 2]
            start, end = _val[0][0], _val[0][1]
            period = end - start
            print(f'found period: {period}')
            remaining_nb_of_iterations = total_iterations - iteration
            remainder_of_period = remaining_nb_of_iterations % period
            for _ in range(remainder_of_period):
                one_cycle(_platform)
            return compute_total_load(_platform)
        one_cycle(_platform)

    assert np.all(_begin_values == np.unique(_platform.as_numpy, return_counts=True)[1])
    return compute_total_load(_platform)


if __name__ == '__main__':
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/14.txt')
    _platform = Grid([list(item) for item in raw_data])

    s1 = solve_1(_platform)
    assert s1 in (108918, 136)
    s2 = solve_2(_platform)
    print(s2)
from functools import reduce
from typing import Callable

import math

import itertools
from collections import defaultdict

from notebooks.aoc_2023.utils import load_stripped_lines


def compute_total_history(_history: list[int]) -> list[list[int]]:
    current_values = _history
    _total_history = [_history]
    while not all(item == 0 for item in current_values):
        # compute diffs
        new_values = [item_2 - item_1 for item_1, item_2 in zip(current_values, current_values[1:])]
        _total_history.append(new_values)
        current_values = _total_history[-1]
        pass
    return _total_history


def solve_1(_histories: list[list[int]], reduce_function: Callable[[int, int], int], idx_element_of_interest: int) -> int:
    next_values_in_history = []
    for history in _histories:
        total_history = compute_total_history(history)
        last_values = [generation[idx_element_of_interest] for generation in total_history[::-1]]
        next_value = reduce(reduce_function, last_values)
        next_values_in_history.append(next_value)
    return sum(next_values_in_history)


if __name__ == "__main__":
    data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/09.txt')
    histories = [[int(item) for item in line.split(' ')] for line in data]

    solution_1 = solve_1(histories, lambda x, y: x + y, -1)
    assert solution_1 == 1861775706
    print(solution_1)

    solution_2 = solve_1(histories, lambda x, y: y - x, 0)
    assert solution_2 == 1082
    print(solution_2)

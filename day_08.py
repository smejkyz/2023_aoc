from functools import reduce

import math

import itertools
from collections import defaultdict

from notebooks.aoc_2023.utils import load_stripped_lines


def load_network(raw_data: list[str]) -> dict:
    _network = defaultdict(dict)
    for line in raw_data[2:]:
        _splitted = line.split(' = ')
        key = _splitted[0]
        values = _splitted[-1][1:-1].split(', ')
        _network[key]['L'] = values[0]
        _network[key]['R'] = values[1]
    return _network


def solve_1(_network, start_position, end_positions: list[str]) -> int:

    current_position = start_position
    for i, direction in enumerate(itertools.cycle(directions)):
        if current_position in end_positions:
            return i
        current_position = _network[current_position][direction]
        pass


def solve_2(_network) -> int:

    starting_positions = [pos for pos in _network.keys() if 'A' in pos]
    ending_positions = [pos for pos in _network.keys() if 'Z' in pos]
    starting_position_hits_z = [solve_1(_network, pos, ending_positions) for pos in starting_positions]
    lcm = reduce(lambda x, y: math.lcm(x, y), starting_position_hits_z)
    return lcm


if __name__ == "__main__":
    data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/08.txt')
    directions = list(data[0])
    network = load_network(data)

    assert solve_1(network, 'AAA', ['ZZZ']) == 12737

    assert solve_2(network) == 9064949303801


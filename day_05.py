from abc import ABC, abstractmethod

import itertools
from collections import defaultdict

from notebooks.aoc_2023.utils import load_stripped_lines

# solve 1 functions:
def iterate_map(id_seed, map):
    for destination_range_start, source_range_start, range_length in map:
        if source_range_start <= id_seed < source_range_start + range_length:
            return id_seed + (destination_range_start - source_range_start)
    return id_seed


def solve_seed(seed: int, maps):
    id_seed = seed
    seed_walk = [seed]
    for map in maps:
        id_seed = iterate_map(id_seed, map)
        seed_walk.append(id_seed)
    print(f'Seed {seed} walk:{seed_walk}')
    return seed_walk[-1]


def solve_1(seeds, maps):
    locations = [solve_seed(seed, maps) for seed in seeds]
    print(min(locations))

def iterate_map_range(id_seed, map):
    for destination_range_start, source_range_start, range_length in map:
        if source_range_start <= id_seed < source_range_start + range_length:
            return id_seed + (destination_range_start - source_range_start)
    return id_seed


def solve_seed_range(seed: list[int, int], maps):
    id_seed = seed
    seed_walk = [seed]
    for map in maps:
        id_seed = iterate_map_range(id_seed, map)
        seed_walk.append(id_seed)
    print(f'Seed {seed} walk:{seed_walk}')
    return seed_walk[-1]


def solve_2(seeds, maps):
    locations = [solve_seed_range(seed, maps) for seed in seeds]
    print(min(locations))


def create_maps(data) -> list:
    blanks = [i for i, item in enumerate(data) if not item] + [len(data)]
    maps = []

    for key_id, (start, end) in enumerate(zip(blanks, blanks[1:])):
        key = data[start + 1]
        values = []
        for i in range(start + 2, end):
            line = data[i].split(' ')
            assert len(line) == 3
            values.append(tuple(int(item) for item in line))
        maps.append(values)
    assert len(maps) == 7
    return maps


# solve 2 using interval/range
SEED = range


class Seeds:
    def __init__(self, raw_values: list[int] = None, part_one: bool = True):

        raw_values = raw_values if raw_values is not None else []
        # todo: this logic should not be done inside __init__  but outside
        if part_one:
            self.values: list[range] = [range(value, value+1) for value in raw_values]
        else:
            seeds_start = raw_values[::2]
            seeds_ranges = raw_values[1::2]

            self.values: list[range] = [range(seed, seed + r) for seed, r in zip(seeds_start, seeds_ranges)]

    def __repr__(self):
        values = [list(_range) for _range in self.values]
        return f'{values}'

    def add(self, seed: list[SEED]):
        self.values.extend(seed)

    def min(self) -> int:
        return min(_val[0] for _val in self.values)


class Transformer(ABC):
    @abstractmethod
    def transform(self, _in: Seeds) -> Seeds:
        pass


class Map(Transformer):
    def __init__(self, _raw_map: list[tuple[int, int, int]]) -> None:

        _raw_map_sorted = sorted(_raw_map, key=lambda x: x[1])

        self.source_ranges = [range(_item[1], _item[1] + _item[2]) for _item in _raw_map_sorted]
        # todo: this is a big refactor issue - only the diff source/target can be hold in memory not the whole target range
        self.destination_ranges = [range(_item[0], _item[0] + _item[2]) for _item in _raw_map_sorted]

        start = self.source_ranges[0][0]
        if start != 0:
            self.source_ranges = [range(0, start)] + self.source_ranges
            self.destination_ranges = [range(0, start)] + self.destination_ranges

        # add last identity
        end = self.source_ranges[-1][-1]
        self.source_ranges = self.source_ranges + [range(end+1, end + 100000000000)]
        self.destination_ranges = self.destination_ranges + [range(end+1, end + 100000000000)]

        # glue holes
        add = []
        for _id, (one, two) in enumerate(zip(self.source_ranges, self.source_ranges[1:])):
            if one[-1] + 1 != two[0]:
                _range = range(one[-1] + 1, two[0])
                add.append((_id, _range))
        for i, (_id, _range) in enumerate(add):
            self.source_ranges.insert(_id+1+i, _range)
            self.destination_ranges.insert(_id+1+i, _range)
        # assert that the source regions covers N:
        for one, two in zip(self.source_ranges, self.source_ranges[1:]):
            assert one[-1] + 1 == two[0]

        pass

    def transform(self, input_seeds: Seeds) -> Seeds:
        output_seed = Seeds()
        for _seed in input_seeds.values:
            output_seed.add(self._transform(_seed))
        return output_seed

    def _transform(self, _seed: SEED) -> list[SEED]:
        if _seed[-1] > self.source_ranges[-1][-1]:
            raise ValueError()
        # first find all source_intervals that have intersection
        intersections = []
        for _id, source in enumerate(self.source_ranges):
            inter = self.compute_intersection(_seed, source)
            if len(inter) > 0:
                intersections.append((_id, source, inter))

        # now we can compute the return value
        output = []
        for _id, source, inter in intersections:
            # todo: this is a big refactor issue - only the diff source/target can be hold in memory not the whole range
            destination_range = self.destination_ranges[_id]
            diff = destination_range[0] - source[0]
            out = range(inter[0] + diff, inter[-1] + diff + 1)
            output.append(out)
        return output

    def has_intersection(self, x: range, y: range) -> bool:
        return len(self.compute_intersection(x, y)) > 0

    @staticmethod
    def compute_intersection(x: range, y: range) -> range:
        return range(max(x[0], y[0]), min(x[-1], y[-1])+1)

import time

if __name__ == "__main__":
    data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/05.txt')
    _seeds = [int(item) for item in data[0].split(':')[1].split( )]

    _maps = create_maps(data)
    # print(solve_1(seeds, maps))

    seeds = Seeds(_seeds, part_one=False)

    maps = [Map(_map) for _map in _maps]

    start = time.time()
    for nb, map in enumerate(maps):
        seeds = map.transform(seeds)
    print('It took', time.time() - start, 'seconds.')

    print(seeds.min())
    pass
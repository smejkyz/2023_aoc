from enum import Enum

import itertools
from tqdm import tqdm

from notebooks.aoc_2023.utils import load_stripped_lines
# https://www.reddit.com/r/adventofcode/comments/18hbbxe/2023_day_12python_stepbystep_tutorial_with_bonus/

def create_new_string(_inds, _characters):
    result = ['.'] * len(_characters)
    for i in range(len(_characters)):
        if i in _inds or _characters[i] == '#':
            result[i] = '#'
    return result


def has_correct_properties(new_string, numbers):
    _tmp = ''.join(new_string)
    _tmp_splitted = [ val for val in _tmp.split('.') if len(val) > 0]
    if len(_tmp_splitted) != len(numbers):
        return False
    for one, two in zip(_tmp_splitted, numbers, strict=True):
        if len(one) != two:
            return False
    return True


def find_combinations(line: str):
    numbers = [int(item) for item in line.split(' ')[1].split(',')]
    characters = list(line.split(' ')[0])
    working = len([char for char in characters if char == '#'])
    needed = sum(numbers) - working

    unknown_positions = [i for i, val in enumerate(characters) if val == '?']

    satisfied_numbers = []
    for inds in itertools.combinations(unknown_positions, needed):
        new_string = create_new_string(inds, characters)
        if has_correct_properties(new_string, numbers):
            satisfied_numbers.append(new_string)
    return len(satisfied_numbers)

ALL_PATH = []

class Result(Enum):
    MUST_BE_EMPTY_SPACE = '.'
    MUST_BE_SPRING = '#'
    CAN_BE_BOTH = '.#'


def next_character_rules(code_so_far, characters, numbers) -> Result:
    if len(code_so_far) == 0:
        return Result.CAN_BE_BOTH
    else:
        joined = ''.join(code_so_far)
        nb_springs = joined.count('#')
        if nb_springs > sum(numbers):
            return Result.MUST_BE_EMPTY_SPACE
        split_by_dot = joined.split('.')
        if len(split_by_dot) == 1:
            # I am still in the first section
            nb_should_have = numbers[0]
            if nb_should_have == nb_springs:
                # next must be '.'
                return Result.MUST_BE_EMPTY_SPACE
            else:
                return Result.MUST_BE_SPRING
        elif len(split_by_dot) == 2:
            # I am in second department
            nb_springs = split_by_dot[1].count('#')
            if nb_springs == numbers[1]:
                return Result.MUST_BE_EMPTY_SPACE
            if code_so_far[-1] == '#':
                return Result.MUST_BE_SPRING
            return Result.CAN_BE_BOTH
        else:
            raise NotImplemented


def fnc(code_so_far: list[str], characters, numbers):
    if sum([char == '#' for char in code_so_far]) == sum(numbers):
        # new configuration found
        ALL_PATH.append(code_so_far)
        return True

    next_symbol = characters[len(code_so_far)]
    if next_symbol in ('.', "#"):
        fnc(code_so_far + [next_symbol], characters, numbers)
    assert next_symbol == '?'
    decision = next_character_rules(code_so_far, characters, numbers)
    if decision == Result.MUST_BE_EMPTY_SPACE:
        fnc(code_so_far + ['.'], characters, numbers)
    elif decision == Result.MUST_BE_SPRING:
        fnc(code_so_far + ['#'], characters, numbers)
        # next in line cannot be spring
    elif decision == Result.CAN_BE_BOTH:
        for symbol in ('#', '.'):
            fnc(code_so_far + [symbol], characters, numbers)
    else:
        raise NotImplemented()


def find_combination_refactor(line: str):
    global ALL_PATH
    numbers = [int(item) for item in line.split(' ')[1].split(',')]
    characters = list(line.split(' ')[0])
    line_as_string = list(line)
    _code = []
    fnc(_code, characters, numbers)


def find_combinations_2(line: str):
    numbers = [int(item) for item in line.split(' ')[1].split(',')]
    characters = list(line.split(' ')[0])

    numbers = 5 * numbers
    characters = characters + ['?'] + characters + ['?'] + characters + ['?'] + characters + ['?'] + characters

    working = len([char for char in characters if char == '#'])
    needed = sum(numbers) - working

    unknown_positions = [i for i, val in enumerate(characters) if val == '?']
    # print(unknown_positions)
    satisfied_numbers = []
    for inds in tqdm(itertools.combinations(unknown_positions, needed)):
        new_string = create_new_string(inds, characters)
        if has_correct_properties(new_string, numbers):
            satisfied_numbers.append(new_string)
    print(f'combinations: {len(satisfied_numbers)}')
    return len(satisfied_numbers)


def solve_2(_data: list[str]):
    result = 0
    for _line in _data:
        print(f'{_line=}')
        base = find_combinations(_line)
        pass
        numbers = _line.split(' ')[1]
        characters = _line.split(' ')[0]

        extended_line = characters + '?' + characters + ' ' + numbers + ',' + numbers
        extended = find_combinations(extended_line)
        sub_total = base * (extended/base) ** 4
        print(sub_total)
        result += sub_total

    return result


if __name__ == "__main__":
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/12_test.txt')
    total = 0
    for line in raw_data:
        combinations = find_combination_refactor(line)
        print(combinations)
        total += combinations
    print(total)
    raise NotImplemented
    #
    # # two:
    # total = 0
    # for line in raw_data:
    #     print(f'{line=}')
    #     combinations = find_combinations_2(line)
    #     print(combinations)
    #     total += combinations
    #
    # print(total)

    line = '?.?.???????##?????? 1,2,8'
    c = find_combinations(line)
    line_ii = '??###??????????###???????? 3,2,1,3,2,1'

    line_1 = '???.###????.### 1,1,3,1,1,3'
    c_line_1 = find_combinations(line_1)

    line_2 = '.??..??...?##.?.??..??...?##. 1,1,3,1,1,3'
    c_line_2 = find_combinations(line_2)

    line_4 = '????.#...#...?????.#...#... 4,1,1,4,1,1'
    c_line_4 = find_combinations(line_4)

    line_5 = '????.######..#####.?????.######..#####. 1,6,5,1,6,5'
    c_line_5 = find_combinations(line_5)

    print(f'total result is: {solve_2(raw_data)}')


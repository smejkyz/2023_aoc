from collections import defaultdict, Counter

from notebooks.aoc_2023.utils import load_stripped_lines


def is_number(str) -> bool:
    try:
        to_int = int(str)
        return True
    except:
        return False


def parse_number_on_line(line: str, line_id: int, interest_symbols: set[str]):
    symbols_as_str = ''.join(interest_symbols)
    numbers = set()
    splitted_by_dot = [item.strip(symbols_as_str) for item in line.split('.') if len(item) > 0]
    for possible_number in splitted_by_dot:
        if len(possible_number) == 0:
            continue
        if is_number(possible_number):
            numbers.add(possible_number)
        else:
            # possible number has to be composed of numbers with symbols between them, find all symbols:
            values = [0] + [i for symbol in interest_symbols for i in range(len(possible_number)) if possible_number.startswith(symbol, i)] + [len(possible_number)]
            for start, end in zip(values, values[1:]):
                _number = possible_number[start:end].strip(symbols_as_str)
                if is_number(_number):
                    numbers.add(_number)
                else:
                    raise ValueError()

    # find possition of each number on the line:
    result: list[tuple[str, int, int]] = []
    for number in numbers:
        number_positions = [i for i in range(len(line)) if line.startswith(number, i) and line[i:i+len(number)] == number]
        number_positions_selected = [i for i in number_positions if i + len(number) == len(line) or line[i+len(number)] not in {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}]
        number_positions_selected = [i for i in number_positions_selected if i == 0 or line[i - 1] not in {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}]
        if not number_positions_selected:
            raise ValueError
        for position in number_positions_selected:
            result.append((number, line_id, position))

    # assert that there is no overlap of numbers
    if len(result) != len(numbers):
        print('')
    return result


def get_adjant_symbols(number, line_id, start_position, data):
    if data[line_id][start_position:start_position+len(number)] != number:
        raise ValueError('')
    symbols_below = set()
    symbols_left_right = set()
    symbols_above = set()

    symbols_start = max(0, start_position - 1)
    symbols_end = min(len(data[line_id])-1, start_position + len(number) + 1)

    if line_id < len(data) - 1:
        line_below = data[line_id + 1][symbols_start:symbols_end]
        symbols_below.update(set(list(line_below)))

    if line_id > 0:
        line_above = data[line_id - 1][symbols_start:symbols_end]
        symbols_above.update(set(list(line_above)))

    if start_position > 0:
        symbols_left_right.add(data[line_id][start_position - 1])
    if start_position + len(number) < len(data[line_id]):
        symbols_left_right.add(data[line_id][start_position + len(number)])

    all_symbols = symbols_above | symbols_below | symbols_left_right

    return all_symbols


def solve_1(data, symbols, parsed_numbers):
    numbers_in_engine = []
    for numbers_on_one_line in parsed_numbers:
        for number, line_id, start_position in numbers_on_one_line:
            adjent_symbols = get_adjant_symbols(number, line_id, start_position , data)
            if len(adjent_symbols & symbols) > 0:
                numbers_in_engine.append(int(number))
    print(sum(numbers_in_engine))
    pass


def get_starts_position_around_number(number, line_id, start_position, data):
    if data[line_id][start_position:start_position+len(number)] != number:
        raise ValueError('')

    symbols_start = max(0, start_position - 1)
    symbols_end = min(len(data[line_id])-1, start_position + len(number) + 1)
    stars_around = []
    if line_id < len(data) - 1:
        line_below = data[line_id + 1][symbols_start:symbols_end]
        if '*' in line_below:
            stars_positions = [(line_id + 1, symbols_start + i) for i in range(len(line_below)) if line_below[i] == '*']
            stars_around.extend(stars_positions)

    if line_id > 0:
        line_above = data[line_id - 1][symbols_start:symbols_end]
        if '*' in line_above:
            stars_positions = [(line_id - 1, symbols_start + i) for i in range(len(line_above)) if line_above[i] == '*']
            stars_around.extend(stars_positions)

    if start_position > 0 and data[line_id][start_position - 1] == '*':
        stars_around.append((line_id, start_position - 1))
    if start_position + len(number) < len(data[line_id]) and data[line_id][start_position + len(number)] == '*':
        stars_around.append((line_id, start_position + len(number)))

    for position in stars_around:
        assert data[position[0]][position[1]] == '*'

    return stars_around

def solve_2(data, symbols, parsed_numbers):
    numbers_in_engine = []
    stars_positions = [(line_id, i) for line_id, line in enumerate(data) for i in range(len(line)) if line[i] == '*']
    result = defaultdict(list)
    for numbers_on_one_line in parsed_numbers:
        for number, line_id, start_position in numbers_on_one_line:
            starts_around = get_starts_position_around_number(number, line_id, start_position , data)
            for position in starts_around:
                assert position in stars_positions
                result[position].append(number)

    gear_ratio = 0
    for key, value in result.items():
        if len(value) == 2:
            gear_ratio += int(value[0]) * int(value[1])

    print(gear_ratio)

if __name__ == "__main__":
    data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/03.txt')
    symbols = set(Counter(''.join(data))) - {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'}
    parsed_numbers = [parse_number_on_line(line, i, symbols) for i, line in enumerate(data)]
    #result_1 = solve_1(data, symbols, parsed_numbers)

    #

    result_2 = solve_2(data, symbols, parsed_numbers)
    pass
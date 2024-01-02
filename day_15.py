from notebooks.aoc_2023.utils import load_stripped_lines


def char_to_ascii(a: str) -> int:
    assert len(a) == 1
    return ord(a)


def decode(_input: str) -> int:
    current_value = 0
    for char in _input:
        current_value += char_to_ascii(char)
        current_value = 17 * current_value
        current_value = current_value % 256
    return current_value


def solve_1(_values: list[str]) -> int:
    _sum = 0
    for _char in _values:
        ind_value = decode(_char)
        _sum += ind_value
        print(f'{_char=}: {ind_value}')
    return _sum


def compute_box_power(bbox_id: int, values: list[tuple[str, int]]):
    _sum = 0
    for id_slot, (_, focus_length) in enumerate(values):
        _sum += (bbox_id + 1) * (id_slot + 1) * focus_length
    return _sum


def solve_2(_values: list[str]) -> int:
    boxes = {i: [] for i in range(256)}
    pass
    for _val in _values:
        print(_val)
        operation_character = '-' if _val[-1] == '-' else '='
        if operation_character == '=':
            label = _val.split(operation_character)[0]
            box_label = decode(label)
            focal_length = int(_val.split(operation_character)[-1])
            # check if the given box has the label:
            indx_present = [i for i, (lb, _) in enumerate(boxes[box_label]) if lb == label]
            if indx_present:
                assert len(indx_present) == 1
                boxes[box_label][indx_present[0]] = (label, focal_length)
            else:
                boxes[box_label].append((label, focal_length))
            pass
        elif operation_character == '-':
            label = _val.split(operation_character)[0]
            box_label = decode(label)
            # check the given box and remove the label if present
            indx_present = [i for i, (lb, _) in enumerate(boxes[box_label]) if lb == label]
            if indx_present:
                assert len(indx_present) == 1
                boxes[box_label].pop(indx_present[0])
            pass
        else:
            raise ValueError
        pass
    focus_powers = [compute_box_power(bbox_id, values) for bbox_id, values in boxes.items()]
    return sum(focus_powers)

if __name__ == '__main__':
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/15.txt')
    individual_values = raw_data[0].split(',')
    s1 = solve_1(individual_values)
    print(s1)
    s2 = solve_2(individual_values)
    print(s2)
    pass
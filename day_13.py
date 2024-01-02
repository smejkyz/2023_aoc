from typing import Any

from notebooks.aoc_2023.utils import load_stripped_lines, load_raw_lines, Grid


def find_patterns(raw_data):
    all_joined = ''.join(raw_data).split('\n\n')
    patterns = [Grid([list(item) for item in block.split('\n')]) for block in all_joined]
    return patterns


def is_reflective_column(reflection_id: tuple[int, int], pattern: Grid) -> bool:
    for id_left, id_right in zip(range(reflection_id[0], -1, -1), range(reflection_id[1], pattern.width, 1)):
        left_column = pattern.get_column(id_left)
        right_column = pattern.get_column(id_right)
        print(f'{left_column=}, {right_column=}')
        if left_column != right_column:
            return False
    return True


def is_reflective_row(reflection_id: tuple[int, int], pattern: Grid) -> bool:
    for id_left, id_right in zip(range(reflection_id[0], -1, -1), range(reflection_id[1], pattern.height, 1)):
        upper_row = pattern.get_row(id_left)
        lower_row = pattern.get_row(id_right)
        print(f'{lower_row==upper_row}')
        if lower_row != upper_row:
            return False
    return True


def is_reflective_row_with_smudge(reflection_id: tuple[int, int], pattern: Grid) -> tuple[bool, Any]:
    smudge_position = None
    for id_left, id_right in zip(range(reflection_id[0], -1, -1), range(reflection_id[1], pattern.height, 1)):
        upper_row = pattern.get_row(id_left)
        lower_row = pattern.get_row(id_right)
        differs_on_position = [u != l for u, l in zip(upper_row, lower_row)]
        if sum(differs_on_position) > 1:
            return False, None

        if sum(differs_on_position) == 1 and smudge_position is not None:
            return False, None

        if sum(differs_on_position) == 1 and smudge_position is None:
            # differs on one position but the is place for correction
            smudge_position = (id_left, differs_on_position.index(True))

    return True, smudge_position


def is_reflective_column_with_smudge(reflection_id: tuple[int, int], pattern: Grid) -> tuple[bool, Any]:
    smudge_position = None
    for id_left, id_right in zip(range(reflection_id[0], -1, -1), range(reflection_id[1], pattern.width, 1)):
        left = pattern.get_column(id_left)
        right = pattern.get_column(id_right)
        differs_on_position = [u != l for u, l in zip(left, right)]
        if sum(differs_on_position) > 1:
            return False, None

        if sum(differs_on_position) == 1 and smudge_position is not None:
            return False, None

        if sum(differs_on_position) == 1 and smudge_position is None:
            # differs on one position but the is place for correction
            smudge_position = (differs_on_position.index(True), id_left)

    return True, smudge_position


def find_vertical_reflection_position(pattern: Grid, part_two: bool) -> tuple[int, int] | None:
    for start in range(pattern.width-1):
        reflection_id = (start, start + 1)
        if part_two:
            is_reflective, smudge = is_reflective_column_with_smudge(reflection_id, pattern)
            if is_reflective and smudge is not None:
                return reflection_id, smudge

        if not part_two and is_reflective_column(reflection_id, pattern):
            return reflection_id
    return None, None


def find_horizontal_reflection_position(pattern: Grid, part_two: bool) -> tuple[int, int] | None:
    for start in range(pattern.height-1):
        reflection_id = (start, start + 1)
        if part_two:
            is_reflective, smudge = is_reflective_row_with_smudge(reflection_id, pattern)
            if is_reflective and smudge is not None:
                return reflection_id, smudge

        if not part_two and is_reflective_row(reflection_id, pattern):
            return reflection_id
    return None, None


def solve_1(patterns: list[Grid]) -> int:
    values = []
    for pattern in patterns:
        vertical_reflection = find_vertical_reflection_position(pattern, part_two=False)
        horizontal_reflection = find_horizontal_reflection_position(pattern, part_two=False)
        assert (vertical_reflection is not None and horizontal_reflection is None) or (vertical_reflection is None and horizontal_reflection is not None)
        if vertical_reflection is not None:
            values.append(vertical_reflection[1])
        if horizontal_reflection is not None:
            values.append(100 * horizontal_reflection[1])
    return sum(values)


def solve_2(patterns: list[Grid]) -> int:
    values = []
    for pattern in patterns:
        vertical_reflection, v_smudge = find_vertical_reflection_position(pattern, part_two=True)
        horizontal_reflection, h_smudge = find_horizontal_reflection_position(pattern, part_two=True)
        assert v_smudge is None or h_smudge is None
        assert v_smudge is not None or h_smudge is not None  # at least one of them is not none
        if v_smudge is None:
            values.append(100 * horizontal_reflection[1])
        if h_smudge is None:
            values.append(vertical_reflection[1])

    return sum(values)


if __name__ == '__main__':
    raw_data = load_raw_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/13.txt')
    _patterns = find_patterns(raw_data)
    #s1 = solve_1(_patterns)
    #assert s1 in (28651, 405)

    s2 = solve_2(_patterns)
    print(s2)
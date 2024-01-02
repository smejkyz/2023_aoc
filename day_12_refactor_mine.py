from functools import cache

from notebooks.aoc_2023.utils import load_stripped_lines


@cache
def fnc(chars: str, vals: list[int]) -> int:
    if not vals:
        if '#' not in chars:
            return 1
        else:
            return 0
    if not chars:
        return 0

    next_char = chars[0]
    next_val = vals[0]

    def dot_function():
        return fnc(chars[1:], vals)

    def sharp_function():
        this_group = chars[:next_val]
        this_group = this_group.replace('?', '#')
        if this_group != next_val * '#':
            return 0

        # done and there's only one possibility
        if len(chars) == next_val:
            # Make sure this is the last group
            if len(vals) == 1:
                # We are valid
                return 1
            else:
                # There's more groups, we can't make it work
                return 0

        if chars[len(this_group)] in ('.', '?'):
            return fnc(chars[len(this_group)+1:], vals[1:])
        # dead end
        return 0

    if next_char == '.':
        out = dot_function()
    elif next_char == '#':
        out = sharp_function()
    elif next_char == '?':
        out = dot_function() + sharp_function()
    else:
        raise ValueError()
    # Help with debugging
    print(chars, vals, "->", out)
    return out


if __name__ == "__main__":
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/12.txt')
    total = 0
    for line in raw_data:
        numbers = [int(item) for item in line.split(' ')[1].split(',')]
        characters = line.split(' ')[0]
        combinations = fnc(characters, tuple(numbers))
        total += combinations
    print(total)

    # solve part II:
    total = 0
    for line in raw_data:
        numbers = [int(item) for item in line.split(' ')[1].split(',')]
        characters = line.split(' ')[0]
        numbers = 5 * numbers
        characters = characters + '?' + characters + '?' + characters + '?' + characters + '?' + characters

        combinations = fnc(characters, tuple(numbers))
        print(combinations)
        total += combinations
    print(total)
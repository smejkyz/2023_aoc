import sys
import functools

from notebooks.aoc_2023.utils import load_stripped_lines


output = 0


@functools.lru_cache(maxsize=None)
def calc(record, groups):
    ## ADD LOGIC HERE ... Base-case logic will go here
    if not groups:
        if '#' not in record:
            return 1
        else:
            return 0
    if not record:
        # invalid
        return 0
    # Look at the next element in each record and group
    next_character = record[0]
    next_group = groups[0]

    # Logic that treats the first character as pound-sign "#"
    def pound():
        # first character is # -
        this_group = record[:next_group]
        this_group = this_group.replace('?', '#')
        if this_group != next_group * '#':
            # not enough space
            return 0

        # If the rest of the record is just the last group, then we're
        # done and there's only one possibility
        if len(record) == next_group:
            # Make sure this is the last group
            if len(groups) == 1:
                # We are valid
                return 1
            else:
                # There's more groups, we can't make it work
                return 0

        # make sure that next separator is valid:
        if record[next_group] in ('?', '.'):
            return calc(record[next_group + 1:], groups[1:])

        #  anything other is not possible
        return 0

    # Logic that treats the first character as dot "."
    def dot():
        #  calc() on a substring - just skip
        return calc(record[1:], groups)

    if next_character == '#':
        # Test pound logic
        out = pound()

    elif next_character == '.':
        # Test dot logic
        out = dot()

    elif next_character == '?':
        # This character could be either character, so we'll explore both
        # possibilities
        out = dot() + pound()

    else:
        raise RuntimeError

    # Help with debugging
    print(record, groups, "->", out)
    return out

if __name__ == '__main__':
    raw_data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/12.txt')
    total = 0
    for line in raw_data:
        numbers = [int(item) for item in line.split(' ')[1].split(',')]
        characters = line.split(' ')[0]
        combinations = calc(characters, tuple(numbers))
        print(combinations)
        total += combinations
    print(total)

    # solve part II:
    total = 0
    for line in raw_data:
        numbers = [int(item) for item in line.split(' ')[1].split(',')]
        characters = line.split(' ')[0]
        numbers = 5 * numbers
        characters = characters + '?' + characters + '?' + characters + '?' + characters + '?' + characters

        combinations = calc(characters, tuple(numbers))
        print(combinations)
        total += combinations
    print(total)


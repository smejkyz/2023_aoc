from functools import reduce

from notebooks.aoc_2023.utils import load_raw_lines, load_stripped_lines


def solve_1(times, distances):
    record_beats = []
    for time, record_distance in zip(times, distances):
        distances_traveled = [hold_time*(time-hold_time) for hold_time in range(time+1)]
        beats_recond = [dist > record_distance for dist in distances_traveled]
        record_beats.append(sum(beats_recond))
    return reduce((lambda x, y: x * y), record_beats)


if __name__ == "__main__":
    data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/06.txt')
    times = [int(item) for item in data[0].split(':')[1].split(' ') if item]
    distances = [int(item) for item in data[1].split(':')[1].split(' ') if item]
    print(solve_1(times, distances))


    times_2 = [int(data[0].split(':')[1].replace(' ', ''))]
    distances_2 = [int(data[1].split(':')[1].replace(' ', ''))]
    print(solve_1(times_2, distances_2))
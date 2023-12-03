from notebooks.aoc_2023.utils import load_lines_as_integers, load_stripped_lines


def solve_1(a: list[str]):
    calibration_digits = []
    for line in a:
        single_line = []
        digits = list(line)
        for d in digits:
            try:
                single_line.append(int(d))
            except:
                pass
        calibration_digits.append(int(str(single_line[0])+str(single_line[-1])))
    return sum(calibration_digits)



    pass


valid_symbols_1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
valid_symbols_2 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

MAP = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
}


def solve_2(a: list[str]):
    calibration_digits = []
    for line in a:
        single_line = []
        for symbol in valid_symbols_2:
            single_line.extend([(i, symbol) for i in range(len(line)) if line.startswith(symbol, i)])

            # if line.find(symbol) != -1:
            #     single_line.append((line.find(symbol), symbol))
            # if line_backwords.find(symbol) != -1:
            #     single_line.append((len(line) - line_backwords.find(symbol), symbol))

        single_line.sort(key=lambda x: x[0])

        first_digit = single_line[0][1]
        last_digit = single_line[-1][1]

        if len(first_digit) > 1:
            first_digit = MAP[first_digit]
        if len(last_digit) > 1:
            last_digit = MAP[last_digit]
        assert len(first_digit) == len(last_digit) == 1
        calibraion = int(first_digit + last_digit)
        calibration_digits.append(calibraion)
    return sum(calibration_digits)


if __name__ == "__main__":
    data = load_stripped_lines('/Users/user/Projects/bp-forjerry-root/projects/bp-experiment/training/notebooks/aoc_2023/input_data/01.txt')

    print(solve_1(data))
    print(solve_2(data))


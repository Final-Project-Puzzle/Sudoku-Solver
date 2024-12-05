import random
import copy
import sys
import multiprocessing
from typing import Dict, Tuple, List, Optional

# backtracking recursion (og number is 100, but to keep it safe using 10000 incase of a complicated initial board)
sys.setrecursionlimit(10000)

#init board generator + valid check + unique check + diff choice (clues up to change)
def generate_sudoku(difficulty='Easy') -> Dict[Tuple[int, int], int]:
    def fill_board():
        board = [[0] * 9 for _ in range(9)]

        def is_valid(num, row, col):
            box_index = (row // 3) * 3 + (col // 3)
            return (num not in rows[row] and
                    num not in cols[col] and
                    num not in boxes[box_index])

        def fill():
            for i in range(9):
                for j in range(9):
                    if board[i][j] == 0:
                        random_nums = list(range(1, 10))
                        random.shuffle(random_nums)
                        for num in random_nums:
                            if is_valid(num, i, j):
                                board[i][j] = num
                                rows[i].add(num)
                                cols[j].add(num)
                                boxes[(i // 3) * 3 + (j // 3)].add(num)

                                if fill():
                                    return True

                                
                                board[i][j] = 0
                                rows[i].remove(num)
                                cols[j].remove(num)
                                boxes[(i // 3) * 3 + (j // 3)].remove(num)
                        return False
            return True

        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]

        fill()
        return board

    def remove_numbers(board, clues):
        positions = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(positions)
        for r, c in positions:
            if board[r][c] == 0:
                continue
            temp = board[r][c]
            board[r][c] = 0

            # uniqueness check
            board_dict = {(i, j): board[i][j] for i in range(9) for j in range(9)}
            if count_solutions(board_dict, 2) != 1:
                board[r][c] = temp  
            else:
                if sum(1 for row in board for num in row if num != 0) <= clues:
                    break

    #NO. of solutions (e.g. nkd triples with initial gird on hard diff will only have 1 solution)
    def count_solutions(board_dict, limit=2):
        solutions_count = 0

        def backtrack():
            nonlocal solutions_count
            if solutions_count >= limit:
                return

            empty_cells = [cell for cell in board_dict if board_dict[cell] == 0]
            if not empty_cells:
                solutions_count += 1
                return

            cell = min(empty_cells, key=lambda c: len(get_possible_values(c, board_dict)))
            for num in get_possible_values(cell, board_dict):
                if is_valid_move(cell, num, board_dict):
                    board_dict[cell] = num
                    backtrack()
                    board_dict[cell] = 0

                    if solutions_count >= limit:
                        return

        def get_possible_values(cell, board_dict):
            row, col = cell
            used = set()
            for k in range(9):
                used.add(board_dict.get((row, k), 0))
                used.add(board_dict.get((k, col), 0))
                used.add(board_dict.get((3 * (row // 3) + k // 3, 3 * (col // 3) + k % 3), 0))
            used.discard(0)
            return [num for num in range(1, 10) if num not in used]

        def is_valid_move(cell, num, board_dict):
            row, col = cell
            for k in range(9):
                if board_dict.get((row, k), 0) == num:
                    return False
                if board_dict.get((k, col), 0) == num:
                    return False
                if board_dict.get((3 * (row // 3) + k // 3, 3 * (col // 3) + k % 3), 0) == num:
                    return False
            return True

        backtrack()
        return solutions_count

    #diff of clues available in each method, can be reduced
    #hint: try to not go below 17 clues as it may take more time to solve esp using nkd pairs/triples
    #hint 2: if going below 17 clues, keep multiprocess on
    #suggestion: run on cmd if trying to use multiprocess as spyder has some limitations
    if difficulty == 'Easy':
        clues = 36
    elif difficulty == 'Medium':
        clues = 32
    elif difficulty == 'Hard':
        clues = 28
    else:
        clues = 28

    # Generate a full board
    filled_board = fill_board()

    
    attempts = 5
    while attempts > 0:
        board_copy = [row[:] for row in filled_board]
        remove_numbers(board_copy, clues)
        board_dict = {(i, j): board_copy[i][j] for i in range(9) for j in range(9)}
        if count_solutions(board_dict, 2) == 1:
            break
        attempts -= 1
    else:
        board_dict = {(i, j): filled_board[i][j] for i in range(9) for j in range(9)}

    return board_dict

#update vals after each step, used in all algorithms
def update_possible_values(cell, num, possible_values_local):
    row, col = cell
    for c in range(9):
        if (row, c) in possible_values_local:
            possible_values_local[(row, c)].discard(num)
    for r in range(9):
        if (r, col) in possible_values_local:
            possible_values_local[(r, col)].discard(num)
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if (r, c) in possible_values_local:
                possible_values_local[(r, c)].discard(num)

#BT -> LOCAL to GLOBAL
def backtracking_worker(num, cell, board_dict_local, rows_local, cols_local,
                        boxes_local, empty_cells_local, possible_values_local, depth,
                        solve_with, find_multiple_solutions, max_parallel_depth,
                        steps, max_depth, techniques_used, solutions):
    row, col = cell
    box_index = (row // 3) * 3 + (col // 3)

    board_dict_local[cell] = num
    rows_local[row].add(num)
    cols_local[col].add(num)
    boxes_local[box_index].add(num)
    empty_cells_local.remove(cell)
    del possible_values_local[cell]

    saved_possible_values = copy.deepcopy(possible_values_local)
    update_possible_values(cell, num, possible_values_local)

    steps.value += 1
    techniques_used['Backtracking'] = techniques_used.get('Backtracking', 0) + 1

    backtracking_parallel_subprocess(board_dict_local, rows_local, cols_local,
                                     boxes_local, empty_cells_local, possible_values_local, depth + 1,
                                     solve_with, find_multiple_solutions, max_parallel_depth,
                                     steps, max_depth, techniques_used, solutions)

# Also move backtracking_parallel_subprocess to the global scope
def backtracking_parallel_subprocess(board_dict_local, rows_local, cols_local,
                                     boxes_local, empty_cells_local, possible_values_local, depth,
                                     solve_with, find_multiple_solutions, max_parallel_depth,
                                     steps, max_depth, techniques_used, solutions):
    max_depth.value = max(max_depth.value, depth)
    if not empty_cells_local:
        # Found a solution
        solution = [[board_dict_local.get((i, j), 0) for j in range(9)] for i in range(9)]
        solutions.append(solution)
        return

    # Choose the cell with the least number of possible values
    cell = min(empty_cells_local, key=lambda c: len(possible_values_local[c]))
    nums = list(possible_values_local[cell])
    random.shuffle(nums)

    row, col = cell
    box_index = (row // 3) * 3 + (col // 3)

    for num in nums:
        if num not in rows_local[row] and num not in cols_local[col] and num not in boxes_local[box_index]:
            # Assign number to the cell
            board_dict_local[cell] = num
            rows_local[row].add(num)
            cols_local[col].add(num)
            boxes_local[box_index].add(num)
            empty_cells_local.remove(cell)
            del possible_values_local[cell]

            # Save possible_values before updating
            saved_possible_values = copy.deepcopy(possible_values_local)
            update_possible_values(cell, num, possible_values_local)

            steps.value += 1
            techniques_used['Backtracking'] = techniques_used.get('Backtracking', 0) + 1

            backtracking_parallel_subprocess(board_dict_local, rows_local, cols_local,
                                             boxes_local, empty_cells_local, possible_values_local, depth + 1,
                                             solve_with, find_multiple_solutions, max_parallel_depth,
                                             steps, max_depth, techniques_used, solutions)

            # Stop if not finding multiple solutions
            if not find_multiple_solutions and len(solutions) > 0:
                return

            # Backtrack
            board_dict_local[cell] = 0
            rows_local[row].remove(num)
            cols_local[col].remove(num)
            boxes_local[box_index].remove(num)
            empty_cells_local.append(cell)
            possible_values_local[cell] = {
                n for n in range(1, 10)
                if n not in rows_local[row] and n not in cols_local[col] and n not in boxes_local[box_index]
            }

            possible_values_local.update(saved_possible_values)

def solve_sudoku(board_dict: Dict[Tuple[int, int], int], solve_with: str,
                 use_backtracking: bool = True,
                 find_multiple_solutions: bool = False,
                 parallel: bool = False,
                 manager=None,
                 steps=None,
                 techniques_used=None,
                 solutions=None,
                 max_depth=None) -> Optional[Tuple[List[List[List[int]]], Dict]]:

    steps.value = 0
    techniques_used.clear()
    solutions[:] = []
    max_depth.value = 0

    
    def initialize_sets():
        rows = {i: set() for i in range(9)}
        cols = {i: set() for i in range(9)}
        boxes = {i: set() for i in range(9)}
        empty_cells = []
        possible_values = {}
        for (i, j), num in board_dict.items():
            if num:
                rows[i].add(num)
                cols[j].add(num)
                boxes[(i // 3) * 3 + (j // 3)].add(num)
            else:
                empty_cells.append((i, j))
        for (i, j) in empty_cells:
            possible_values[(i, j)] = {
                num for num in range(1, 10)
                if (num not in rows[i] and num not in cols[j] and num not in boxes[(i // 3) * 3 + (j // 3)])
            }
        return rows, cols, boxes, empty_cells, possible_values

    #initializing
    rows, cols, boxes, empty_cells, possible_values = initialize_sets()

    # Validity check
    def is_valid(num, row, col):
        return (num not in rows[row] and
                num not in cols[col] and
                num not in boxes[(row // 3) * 3 + (col // 3)])

    
    def naked_singles():
        nonlocal possible_values, rows, cols, boxes, empty_cells
        changed = False
        singles = [cell for cell in possible_values if len(possible_values[cell]) == 1]
        if not singles:
            return False
        for cell in singles:
            num = possible_values[cell].pop()
            i, j = cell
            board_dict[cell] = num
            rows[i].add(num)
            cols[j].add(num)
            boxes[(i // 3) * 3 + (j // 3)].add(num)
            empty_cells.remove(cell)
            del possible_values[cell]
            update_possible_values(cell, num, possible_values)
            steps.value += 1
            techniques_used['Naked Singles'] = techniques_used.get('Naked Singles', 0) + 1
            changed = True
        return changed

    # Naked Pairs
    def naked_pairs():
        nonlocal possible_values
        changed = False
        units = []

        # Rows
        for i in range(9):
            units.append([(i, j) for j in range(9)])

        # Columns
        for j in range(9):
            units.append([(i, j) for i in range(9)])

        # Boxes
        for box_row in range(3):
            for box_col in range(3):
                units.append([
                    (box_row * 3 + r, box_col * 3 + c) for r in range(3) for c in range(3)
                ])

        for unit in units:
            # Find cells in the unit that have exactly two possible values
            candidates = [(cell, possible_values[cell]) for cell in unit if cell in possible_values and len(possible_values[cell]) == 2]
            # Look for naked pairs
            for i in range(len(candidates)):
                cell1, vals1 = candidates[i]
                for j in range(i + 1, len(candidates)):
                    cell2, vals2 = candidates[j]
                    if vals1 == vals2:
                        # Found a naked pair
                        for cell in unit:
                            if cell != cell1 and cell != cell2 and cell in possible_values:
                                before = len(possible_values[cell])
                                possible_values[cell] -= vals1
                                after = len(possible_values[cell])
                                if before != after:
                                    changed = True
                                    steps.value += 1
                                    techniques_used['Naked Pairs'] = techniques_used.get('Naked Pairs', 0) + 1
        return changed

    # Naked Triples
    def naked_triples():
        nonlocal possible_values
        changed = False
        units = []

        # Rows
        for i in range(9):
            units.append([(i, j) for j in range(9)])

        # Columns
        for j in range(9):
            units.append([(i, j) for i in range(9)])

        # Boxes
        for box_row in range(3):
            for box_col in range(3):
                units.append([
                    (box_row * 3 + r, box_col * 3 + c) for r in range(3) for c in range(3)
                ])

        for unit in units:
            # Find cells with 2 or 3 possible values
            candidates = [(cell, possible_values[cell]) for cell in unit if cell in possible_values and 2 <= len(possible_values[cell]) <= 3]
            # Look for naked triples
            for i in range(len(candidates)):
                cell1, vals1 = candidates[i]
                for j in range(i + 1, len(candidates)):
                    cell2, vals2 = candidates[j]
                    for k in range(j + 1, len(candidates)):
                        cell3, vals3 = candidates[k]
                        union_vals = vals1 | vals2 | vals3
                        if len(union_vals) == 3:
                            # Found a naked triple
                            for cell in unit:
                                if cell not in [cell1, cell2, cell3] and cell in possible_values:
                                    before = len(possible_values[cell])
                                    possible_values[cell] -= union_vals
                                    after = len(possible_values[cell])
                                    if before != after:
                                        changed = True
                                        steps.value += 1
                                        techniques_used['Naked Triples'] = techniques_used.get('Naked Triples', 0) + 1
        return changed

    
    def constraint_propagation():
        nonlocal empty_cells
        progress = True
        while progress:
            progress = False
            if naked_singles():
                progress = True
                continue
            if solve_with in ('naked_pairs', 'naked_triples'):
                if naked_pairs():
                    progress = True
                if solve_with == 'naked_triples' and naked_triples():
                    progress = True
        return not empty_cells

    #BT paralle algorithms 
    def backtracking_parallel(depth=0, max_parallel_depth=2):
        nonlocal rows, cols, boxes, empty_cells, possible_values
        max_depth.value = max(max_depth.value, depth)
        if not empty_cells:
            
            solution = [[board_dict.get((i, j), 0) for j in range(9)] for i in range(9)]
            solutions.append(solution)
            return

        # Choose the cell with the least number of possible values
        cell = min(empty_cells, key=lambda c: len(possible_values[c]))
        nums = list(possible_values[cell])
        random.shuffle(nums)  # Randomize the order of numbers

        row, col = cell
        box_index = (row // 3) * 3 + (col // 3)

        
        if parallel and depth < max_parallel_depth:
            processes = []
            for num in nums:
                if is_valid(num, row, col):
                    p = multiprocessing.Process(target=backtracking_worker,
                                                args=(num, cell, copy.deepcopy(board_dict),
                                                      copy.deepcopy(rows), copy.deepcopy(cols),
                                                      copy.deepcopy(boxes), empty_cells.copy(),
                                                      copy.deepcopy(possible_values), depth,
                                                      solve_with, find_multiple_solutions, max_parallel_depth,
                                                      steps, max_depth, techniques_used, solutions))
                    processes.append(p)
                    p.start()
            for p in processes:
                p.join()
                
                if not find_multiple_solutions and len(solutions) > 0:
                    for proc in processes:
                        proc.terminate()
                    return
        else:
            for num in nums:
                if is_valid(num, row, col):
                    
                    board_dict[cell] = num
                    rows[row].add(num)
                    cols[col].add(num)
                    boxes[box_index].add(num)
                    empty_cells.remove(cell)
                    del possible_values[cell]

                
                    saved_possible_values = copy.deepcopy(possible_values)
                    update_possible_values(cell, num, possible_values)

                    steps.value += 1
                    techniques_used['Backtracking'] = techniques_used.get('Backtracking', 0) + 1

                    backtracking_parallel(depth + 1, max_parallel_depth)

                    
                    if not find_multiple_solutions and len(solutions) > 0:
                        return

                    
                    board_dict[cell] = 0
                    rows[row].remove(num)
                    cols[col].remove(num)
                    boxes[box_index].remove(num)
                    empty_cells.append(cell)
                    possible_values[cell] = {
                        n for n in range(1, 10)
                        if n not in rows[row] and n not in cols[col] and n not in boxes[box_index]
                    }

                    possible_values.update(saved_possible_values)

    
    if constraint_propagation():
        solution = [[board_dict.get((i, j), 0) for j in range(9)] for i in range(9)]
        solutions.append(solution)
    elif use_backtracking:
        backtracking_parallel()
    else:
        print("No solution found.")
        return None

    "Complexity test" # i can add plot plotting the result if needed
    complexity_report = {
        'Total Steps': steps.value,
        'Techniques Used': dict(techniques_used),
        'Max Backtracking Depth': max_depth.value,
        'Estimated Difficulty': estimate_difficulty(steps.value, techniques_used, max_depth.value)
    }

    #complexity on the BT algorithm
    return list(solutions), complexity_report

def estimate_difficulty(steps, techniques_used, max_depth):
    if techniques_used.get('Backtracking', 0) > 100 or max_depth > 50:
        difficulty = 'Hard'
    elif techniques_used.get('Backtracking', 0) > 0:
        difficulty = 'Medium'
    else:
        difficulty = 'Easy'
    return difficulty

def print_board(board):
    print("\nSudoku Board:")
    for i in range(9):
        row = ''
        for j in range(9):
            num = board[i][j]
            if num == 0:
                row += '. '
            else:
                row += str(num) + ' '
            if j % 3 == 2 and j != 8:
                row += '| '
        print(row.strip())
        if i % 3 == 2 and i != 8:
            print('-' * 21)

#read .txt file
def load_puzzle_from_file(filename):
    board_dict = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            nums = line.strip().split()
            for j, num in enumerate(nums):
                if num == '.':
                    board_dict[(i, j)] = 0
                else:
                    board_dict[(i, j)] = int(num)
    return board_dict

#inital board entered by the user
def input_puzzle_manually():
    print("Enter the puzzle row by row. Use '.' for empty cells.")
    board_dict = {}
    for i in range(9):
        while True:
            row_input = input(f"Row {i+1}: ").strip().split()
            if len(row_input) == 9:
                break
            else:
                print("Please enter exactly 9 numbers.")
        for j, num in enumerate(row_input):
            if num == '.':
                board_dict[(i, j)] = 0
            else:
                board_dict[(i, j)] = int(num)
    return board_dict


def main():
    
    manager = multiprocessing.Manager()
    steps = manager.Value('i', 0)
    techniques_used = manager.dict()
    solutions = manager.list()
    max_depth = manager.Value('i', 0)  #maximum depth of BT recursion

    #User Choice
    while True:
        print("\nChoose Puzzle Source:")
        print("1. Generate Random Puzzle")
        print("2. Input Puzzle Manually")
        print("3. Load Puzzle from File")

        user_input = input("Enter Your Choice: ").strip()
        if user_input in {'1', '2', '3'}:
            if user_input == '1':
                # Dif lvl of generated initial board
                while True:
                    print("\nSelect Difficulty Level:")
                    print("1. Easy")
                    print("2. Medium")
                    print("3. Hard")

                    difficulty_input = input("Enter Your Choice: ").strip()
                    if difficulty_input in {'1', '2', '3'}:
                        if difficulty_input == '1':
                            difficulty = 'Easy'
                        elif difficulty_input == '2':
                            difficulty = 'Medium'
                        elif difficulty_input == '3':
                            difficulty = 'Hard'
                        break
                    else:
                        print("Invalid input. Please enter 1, 2, or 3.")
                sudoku_board_dict = generate_sudoku(difficulty)
                break
            elif user_input == '2':
                sudoku_board_dict = input_puzzle_manually()
                break
            elif user_input == '3':
                filename = input("Enter the filename: ").strip()
                try:
                    sudoku_board_dict = load_puzzle_from_file(filename)
                except FileNotFoundError:
                    print(f"File '{filename}' not found.")
                    continue
                break
        else:
            print("Invalid input. Please enter 1, 2, or 3.")

    #initial board
    initial_board = [[sudoku_board_dict.get((i, j), 0) for j in range(9)] for i in range(9)]
    print_board(initial_board)

    #BT ON/OFF
    while True:
        user_input = input("\nEnable backtracking if constraint propagation fails? (0 for off, 1 for on): ").strip()
        if user_input in {'0', '1'}:
            use_backtracking = user_input == '1'
            break
        else:
            print("Invalid input. Please enter either 0 or 1.")

    #Solving method
    while True:
        print("\nChoose Solving Method:")
        print("1. Naked Singles Only")
        print("2. Naked Pairs Only")
        print("3. Naked Triples Only")

        user_input = input("Enter Your Choice: ").strip()
        if user_input in {'1', '2', '3'}:
            if user_input == '1':
                solve_with = 'naked_singles'
            elif user_input == '2':
                solve_with = 'naked_pairs'
            elif user_input == '3':
                solve_with = 'naked_triples'
            break
        else:
            print("Invalid input. Please enter 1, 2, or 3.")

    # User's choice for finding multiple solutions
    while True:
        user_input = input("\nFind all possible solutions? (0 for No, 1 for Yes): ").strip()
        if user_input in {'0', '1'}:
            find_multiple_solutions = user_input == '1'
            break
        else:
            print("Invalid input. Please enter 0 or 1.")

    # Additional user input for enabling parallel processing
    while True:
        user_input = input("\nEnable parallel processing? (0 for off, 1 for on): ").strip()
        if user_input in {'0', '1'}:
            parallel = user_input == '1'
            break
        else:
            print("Invalid input. Please enter 0 or 1.")

    # Solve the puzzle with the selected method
    result = solve_sudoku(sudoku_board_dict.copy(), solve_with, use_backtracking,
                          find_multiple_solutions, parallel,
                          manager=manager,
                          steps=steps,
                          techniques_used=techniques_used,
                          solutions=solutions,
                          max_depth=max_depth)
    if result:
        solutions_list, complexity_report = result
        if find_multiple_solutions:
            print(f"\nTotal solutions found: {len(solutions_list)}")
            for idx, solution in enumerate(solutions_list, 1):
                print(f"\nSolution {idx}:")
                print_board(solution)
        else:
            print("\nSolved Sudoku Puzzle:")
            print_board(solutions_list[0])

        # Display complexity analysis
        print("\nComplexity Analysis:")
        print(f"Total Steps: {complexity_report['Total Steps']}")
        print(f"Techniques Used: {complexity_report['Techniques Used']}")
        print(f"Max Backtracking Depth: {complexity_report['Max Backtracking Depth']}")
        print(f"Estimated Difficulty: {complexity_report['Estimated Difficulty']}")
    else:
        print("Failed to solve the Sudoku puzzle.")

if __name__ == '__main__':
    main()

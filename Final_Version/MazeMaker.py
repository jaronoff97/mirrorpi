import random
import numpy as np
import cv2
import time

num_rows = 100  # int(input("Rows: "))  # number of rows
num_cols = 100  # int(input("Columns: "))  # number of columns
maze = np.zeros((num_rows, num_cols, 5), dtype=np.uint8)
r = num_rows / 2
c = num_cols / 2
history = [(r, c)]


def generate_maze(history, column, row, image):
    while history:
        # start = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        maze[row, column, 4] = 1  # designate this location as visited
        # check if the adjacent cells are valid for moving to
        check = []
        if column > 0 and maze[row, column - 1, 4] == 0:
            check.append('L')
        if row > 0 and maze[row - 1, column, 4] == 0:
            check.append('U')
        if column < num_cols - 1 and maze[row, column + 1, 4] == 0:
            check.append('R')
        if row < num_rows - 1 and maze[row + 1, column, 4] == 0:
            check.append('D')

        if len(check):  # If there is a valid cell to move to.
            # mazeark the walls between cells as open if we move
            history.append([row, column])
            move_direction = random.choice(check)
            if move_direction == 'L':
                maze[row, column, 0] = 1
                column = column - 1
                maze[row, column, 2] = 1
            if move_direction == 'U':
                maze[row, column, 1] = 1
                row = row - 1
                maze[row, column, 3] = 1
            if move_direction == 'R':
                maze[row, column, 2] = 1
                column = column + 1
                maze[row, column, 0] = 1
            if move_direction == 'D':
                maze[row, column, 3] = 1
                row = row + 1
                maze[row, column, 1] = 1
        else:  # If there are no valid cells to move to.
            # retrace one step back in history if no move is possible
            row, column = history.pop()
        make_image(row, column, image)
        # end = time.time()
        # print(end - start)


# Open the walls at the start and finish


def make_image(row, col, image):
    # Generate the image for display
    cell_data = maze[row, col]
    row_in_image = range(10 * row + 1, 10 * row + 9)
    col_in_image = range(10 * col + 1, 10 * col + 9)
    # print(cell_data)
    for i in row_in_image:
        if cell_data[4] == 0:
            image[i, col_in_image] = 255
            return
        image[i, col_in_image] = 255
        if cell_data[0] == 1:
            image[row_in_image, 10 * col] = 255
        if cell_data[1] == 1:
            image[10 * row, col_in_image] = 255
        if cell_data[2] == 1:
            image[row_in_image,
                  10 * col + 9] = 255
        if cell_data[3] == 1:
            image[10 * row + 9,
                  col_in_image] = 255
    cv2.imshow("Image", image)


def final_show(img):
    for row in range(0, num_rows):
        for col in range(0, num_cols):
            cell_data = maze[row, col]
            for i in range(10 * row + 1, 10 * row + 9):
                img[i, range(10 * col + 1, 10 * col + 9)] = 255
                if cell_data[0] == 1:
                    img[range(10 * row + 1, 10 * row + 9), 10 * col] = 255
                if cell_data[1] == 1:
                    img[10 * row, range(10 * col + 1, 10 * col + 9)] = 255
                if cell_data[2] == 1:
                    img[range(10 * row + 1, 10 * row + 9), 10 * col + 9] = 255
                if cell_data[3] == 1:
                    img[10 * row + 9, range(10 * col + 1, 10 * col + 9)] = 255


def generate_maze_helper(image):
    generate_maze(history, c, r, image)


def makeBlankImage():
    return np.zeros((num_rows * 10, num_cols * 10), dtype=np.uint8)


def main():
    image = makeBlankImage()
    generate_maze(history, c, r, image)
    final_image = makeBlankImage()
    final_show(final_image)
    cv2.imshow("Image", final_image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return

if __name__ == '__main__':
    main()

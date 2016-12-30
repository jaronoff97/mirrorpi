import random
import numpy as np
import cv2
import time
from math import sqrt

num_rows = 36  # int(input("Rows: "))  # number of rows
num_cols = 36  # int(input("Columns: "))  # number of columns
size = (1000, 1000)
maze = np.zeros((num_rows, num_cols, 5), dtype=np.uint8)
range_of_color = (104, 255)
cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)


def make_incrementer(start, up=True):

    def closure(start=start, direction=up):
        while True:
            yield start
            start = start + 1 if direction else start - 1
            if start >= 255 or start <= 104:
                direction = (not direction)
    return closure

color_counter = make_incrementer(104)()


def generate_maze(image, maze, history, column, row):
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
            make_image(image, maze, row, column)
        # end = time.time()
        # print(end - start)


# Open the walls at the start and finish


def make_image(image, maze, row, col):
    # Generate the image for display
    cell_data = maze[row, col]
    row_in_image = range(10 * row + 1, 10 * row + 9)
    col_in_image = range(10 * col + 1, 10 * col + 9)
    # print(cell_data)
    for i in row_in_image:
        color = color_counter.next()
        if cell_data[4] == 0:
            image[i, col_in_image] = color
            return
        image[i, col_in_image] = color
        if cell_data[0] == 1:
            image[row_in_image, 10 * col] = color
        if cell_data[1] == 1:
            image[10 * row, col_in_image] = color
        if cell_data[2] == 1:
            image[row_in_image,
                  10 * col + 9] = color
        if cell_data[3] == 1:
            image[10 * row + 9,
                  col_in_image] = color
    img_to_show = cv2.resize(image, ((size[0]),
                                     (size[1])),
                             interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Image", img_to_show)


def make_final_image(maze, img):
    col_range = ((num_cols / 2 - 1), (num_cols / 2 + 1))
    row_range = ((num_rows / 2 - 1), (num_rows / 2 + 1))
    while (col_range[0] >= 0 and
           col_range[1] <= num_cols and
           row_range[0] >= 0 and
           row_range[1] <= num_rows):
        for row in range(row_range[0], row_range[1]):
            for col in range(col_range[0], col_range[1]):
                if ((row_range[0] + 1 < row < row_range[1] - 1) and
                        (col_range[0] + 1 < col < col_range[1] - 1)):
                    continue
                cell_data = maze[row, col]
                cur_row = range(10 * row + 1, 10 * row + 9)
                cur_col = range(10 * col + 1, 10 * col + 9)
                for i in cur_row:
                    img[i, cur_col] = 255
                    if cell_data[0] == 1:
                        img[cur_row, 10 * col] = 255
                    if cell_data[1] == 1:
                        img[10 * row, cur_col] = 255
                    if cell_data[2] == 1:
                        img[cur_row, 10 * col + 9] = 255
                    if cell_data[3] == 1:
                        img[10 * row + 9, cur_col] = 255
                image_to_show = cv2.resize(img, ((size[0]),
                                                 (size[1])),
                                           interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Image", image_to_show)
        col_range = (col_range[0] - 1, col_range[1] + 1)
        row_range = (row_range[0] - 1, row_range[1] + 1)
        cv2.waitKey(100)


def black_out(img):
    cur_pos = (size[0] / 2, size[1] / 2)
    radius = 1
    velocity = 2
    go_out = True
    while radius < int(sqrt(((size[0] / 2) ** 2) + ((size[1] / 2) ** 2))):
        image_copy = img.copy()
        cv2.circle(image_copy, cur_pos, radius, 0, thickness=-1)
        if radius + velocity < 0:
            continue
        radius += velocity
        velocity = velocity + 2 if go_out else velocity - 1
        if velocity >= 10:
            go_out = False
        elif velocity <= -5:
            go_out = True
        cv2.waitKey(100)
        cv2.imshow("Image", image_copy)
        # print(radius)


def main():
    r = num_rows / 2
    c = num_cols / 2
    while True:
        image = np.zeros((num_rows * 10, num_cols * 10), dtype=np.uint8)
        maze = np.zeros((num_rows, num_cols, 5), dtype=np.uint8)
        history = [(r, c)]
        generate_maze(image, maze, history, c, r)
        make_final_image(maze, image)
        image = cv2.resize(image, ((size[0]),
                                   (size[1])),
                           interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Image", image)

        cv2.waitKey(5000)
        black_out(image)

if __name__ == '__main__':
    main()

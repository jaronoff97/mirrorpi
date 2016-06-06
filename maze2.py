import numpy as np
import cv2
from random import randint, choice

go = True
lines = []
size = (1000, 1000)
next_spot = (size[0] / 2, size[1] / 2)
img = np.zeros((size[0], size[1], 3), np.uint8)


def makeLine(start, end, thickness):
    def direction(self):
        s, e = self["start"], self["end"]
        if s[0] > e[0]:
            return "W"
        elif s[0] < e[0]:
            return "E"
        elif s[1] > e[1]:
            return "S"
        elif s[1] < e[1]:
            return "N"
    line = {
        "start": start,
        "end": end,
        "color": (255, 0, 0),
        "thickness": thickness,
        "direction": direction
    }
    return line


def draw(img):
    for line in lines:
        cv2.line(img, line["start"], line["end"],
                 line["color"], line["thickness"])
    cv2.circle(img, next_spot, 20, (0, 0, 255))
    cv2.imshow("Image", img)


def contains(element, params=["start", "end"]):
    def contains_params(lhs, rhs):
        toReturn = True
        for param in params:
            toReturn = (False if lhs[param] != rhs[param] else True)
        return toReturn
    return (True if len(
        filter(
            lambda x: contains_params(x, element), lines)
        ) != 0 else False)


def generate_v2():
    grid = np.full(size, 1)
    maze = []

    def get_surrounding(point):
        return [(x, y) for x in list(range(point[0] - 1, point[0] + 2))
                for y in list(range(point[1] - 1, point[1] + 2)) if (x, y) != point]
    start_point = (size[0] / 2, size[1] / 2)
    maze.append(start_point)
    surrounding = get_surrounding(start_point)
    while len(surrounding) > 0:
        rand_cell = choice(surrounding)
        neighbors = get_surrounding(rand_cell)
        if len(filter(lambda x: grid[x] != 0, neighbors)) < 2:
            grid[rand_cell] = 0
            surrounding.append(neighbors)
        surrounding.remove(rand_cell)
    print(surrounding)


def legal(test_line):
    for line in lines:
        if line['direction'](line) == test_line['direction'](test_line):
            pass


def generate_v1(initial_spot, line_size=15):

    def get_direction(l_s=line_size):
        direction = randint(0, 3)
        if direction == 0:
            return lambda (x, y): (x + l_s, y) if (not (x > size[0] or x < 0) and not (y > size[1] or y < 0)) else (x - l_s * 5, y)
        elif direction == 1:
            return lambda (x, y): (x - l_s, y) if (not (x > size[0] or x < 0) and not (y > size[1] or y < 0)) else (x + l_s * 5, y)
        elif direction == 2:
            return lambda (x, y): (x, y + l_s) if (not (x > size[0] or x < 0) and not (y > size[1] or y < 0)) else (x, y - l_s * 5)
        elif direction == 3:
            return lambda (x, y): (x, y - l_s) if (not (x > size[0] or x < 0) and not (y > size[1] or y < 0)) else (x, y + l_s * 5)
    line = makeLine(initial_spot, get_direction()(initial_spot), 1)
    print(line)
    if not contains(line):
        lines.append(line)
    return get_direction()(initial_spot)

while go:
    img = np.zeros((size[0], size[1], 3), np.uint8)
    draw(img)
    next_spot = generate_v1(next_spot, 20)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()

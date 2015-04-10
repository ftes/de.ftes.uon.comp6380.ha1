__author__ = 'Fredrik'

import math

def print_set(set):
    for i in set:
        print('%8.5f\t%8.5f\t%3.1f' % (i[0], i[1], i[2]))

import numpy as np
def generate_set(density, max_radius):
    points = 96 * density
    result = np.zeros((points * 4, 3))
    for i in range(0, points):
        angle = i * math.pi / ( 16. * density )
        radius = max_radius * (104 * density - i) / ( 104. * density)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        result[i*4] = [x, y, 0]
        result[i*4+1] = [-x, -y, 1]

        angle += math.pi / 2
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        result[i*4+2] = [x, y, 2]
        result[i*4+3] = [-x, -y, 3]

    return result

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import where
def plot_set(set):
    figure(1)
    ioff()  # interactive graphics off
    clf()   # clear the plot
    hold(True) # overplot on
    for c in range(0,4):
        here = where(set[:,2]==c)
        colors = ['blue', 'red', 'green', 'yellow']
        plot(set[here,0],set[here,1],'o', color=colors[c])
    ion()   # interactive graphics on
    draw()  # update the plot
    ioff()
    show()


set = generate_set(2, 1)
print_set(set)
plot_set(set)

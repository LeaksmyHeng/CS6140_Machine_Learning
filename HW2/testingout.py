import csv
import matplotlib.pyplot as plt

import matplotlib
import numpy as np


def read_csv():
    """
    read csv file that contains all the data and draw a scatter plot with green for data that has y as 1 an red for data that has y as 0,
    @param csv_file: data file in form of csv format
    @return dict_val: dictionary value in form of {1: [[x value], [y-value]], 0: [[x value], [y-value]]}
    """
    dict_val = {'label': [], 'x_val': [], 'y_val': []}
    with open(r'C:\Users\Leaksmy Heng\Documents\GitHub\cs6140\HW2\hw2_data.txt') as file:
        for row in file.readlines():
            row = row.split(',')
            label = int(row[2][0])
            x_val = float(row[0])
            y_val = float(row[1])

            dict_val['label'].append(label)
            dict_val['x_val'].append(x_val)
            dict_val['y_val'].append(y_val)

        return dict_val


def draw_scatter_plot(dict_value):
    """
    Draw scatter plot using matplotlib.
    @param: dict_value containing x_array, y_array and label
    """
    dict_label = dict_value['label']
    dict_x = dict_value['x_val']
    dict_y = dict_value['y_val']
    colors = ['blue', 'red']

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(dict_x, dict_y, c=dict_label, cmap=matplotlib.colors.ListedColormap(colors))

    cb = plt.colorbar()
    loc = np.arange(0, max(dict_label), max(dict_label) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(colors)
    plt.show()
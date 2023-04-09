import logging
import matplotlib.pyplot as plt
import numpy as np
from GDA import GDA

logging.basicConfig(filename='plot_decision_boundary.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def plot_decision_boundary(model, x, y):
    """
    Creating a dicision boundary based on the model provided its x and y.
    :param model: GDA model
    :param x: the input value with the value of [[x1, y1], [x2, y2], [x3, y3], ..., [xn, yn]]
    :type x: list of list
    :param y: label based of the input value. y is in {1, 0}. y is shown as [y1, y2, y3, ... yn)
    :return: list
    """
    # Find the min value for x and y
    # -3 & +3 so that our plot is wider
    x_min = min([point[0] for point in x]) - 3
    y_min = min([point[1] for point in x]) - 3

    # Find the max value for x and y coordinate
    x_max = max([point[0] for point in x]) + 3
    y_max = max([point[1] for point in x]) + 3

    # create a meshgrid based of our x_min, y_min, x_max, y_max with with 0.1 step (element spacing)
    # this will return the coordinate matrices from coordinate vectors.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    logging.info(f'x_min is {x_min}\nx_max is {x_max}\ny_min is {y_min}\ny_max is {y_max}')

    # use zip function to get a tuple of (x_grid, y_grid)
    zip_x_y = zip(xx.ravel(), yy.ravel())
    class_prediction = []
    for x_grid, y_grid in zip_x_y:
        class_prediction.append([model.predict([x_grid, y_grid])])
    # calculate the class prediction for each point in the meshgrid
    class_prediction = np.array(class_prediction)
    class_prediction = class_prediction.reshape(xx.shape)

    # Plot the decision boundary and circular contour
    plt.contour(xx, yy, class_prediction, colors='orange')
    mu0, mu1 = model.mu0, model.mu1
    sigma = model.sigma
    # compute eigen value
    w, v = np.linalg.eig(sigma)
    k = 2.4477  # value for 99% confidence interval for a 2-dimensional distribution

    max_eigan_val = w[1]
    logging.info(f'Eigan value is {w}')

    while True:
        # substract with 0.3 so that we get lots of contour line
        sub_eigan_val = max_eigan_val - 0.3
        if sub_eigan_val < 0:
            break

        ellipse1 = plt.Circle((mu0[0], mu0[1]), k * np.sqrt(max_eigan_val), fill=False, color='blue')
        ellipse2 = plt.Circle((mu0[0], mu0[1]), k * np.sqrt(sub_eigan_val), fill=False, color='blue')
        plt.gca().add_artist(ellipse1)
        plt.gca().add_artist(ellipse2)
        ellipse1 = plt.Circle((mu1[0], mu1[1]), k * np.sqrt(max_eigan_val), fill=False, color='red')
        ellipse2 = plt.Circle((mu1[0], mu1[1]), k * np.sqrt(sub_eigan_val), fill=False, color='red')
        plt.gca().add_artist(ellipse1)
        plt.gca().add_artist(ellipse2)
        max_eigan_val = sub_eigan_val

    # Plot the data points
    plt.scatter([point[0] for point in x if y[x.index(point)] == 0],
                [point[1] for point in x if y[x.index(point)] == 0],
                )
    plt.scatter([point[0] for point in x if y[x.index(point)] == 1],
                [point[1] for point in x if y[x.index(point)] == 1],
                )

    plt.show()


# Generate random data
from testingout import read_csv
dict_value = read_csv()
dict_label = dict_value['label']
dict_x = dict_value['x_val']
dict_y = dict_value['y_val']
x = []
for i in range(len(dict_x)):
    x.append([dict_x[i], dict_y[i]])


# Train GDA model
model = GDA()
model.train(x, dict_label)

# Plot the decision boundary and contour
plot_decision_boundary(model, x, dict_label)
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# orthogonal_regression_implementation.py
#
# Author: Alexander Paul
# Date: 22 JUL 22
# Class: MEDP 7098
#
# Description:
# This python script provides a basic implementation of orthogonal regression
# on a given data set. The 'fit' function calls the scipy iterative
# minimization algorithm and applies it to the perpendicular distance function.
# ----------------------------------------------------------------------------

from numpy import sqrt, linspace, random
from scipy.optimize import minimize


def fit(x_input, y_input, m_start=0, b_start=0):
    global X_ORTHOG_FIT
    global Y_ORTHOG_FIT
    X_ORTHOG_FIT = x_input
    Y_ORTHOG_FIT = y_input
    starting_parameters = [m_start, b_start]
    result = minimize(PerpendicularResiduals, starting_parameters)
    slope = result.x[0]
    y_intercept = result.x[1]
    print("slope = ", slope, " b = ", y_intercept)
    return [slope, y_intercept]


def PerpendicularResiduals(a):
    m_local = a[0]
    b_local = a[1]
    return sum(
        [
            abs(yi - (b_local + m_local * xi)) / sqrt(1 + (m_local ** 2))
            for xi, yi in zip(X_ORTHOG_FIT, Y_ORTHOG_FIT)
        ]
    )


if __name__ == "__main__":
    # Define the true values of the x and y data
    xtrue = linspace(0, 100)
    x = xtrue
    ytrue = [0.6 * xi for xi in xtrue]

    # Add noise to y-values.
    noise_stddev = 5
    noise = random.normal(0, noise_stddev, len(ytrue))
    y = ytrue + noise

    answer = fit(x, y)

    # y = [0.6*xi+20*random.randint(-1,1)*random.random() for xi in x]

    print("Results:\n\t m = ", answer[0], "\n\t b = ", answer[1])

import matplotlib.pyplot as plt
import numpy as np
import math

def calculate_average_x(list_points, p):
    list_of_points = list(sum(j for j in i[:1]) for i in list_points) if p == 'x' else list(sum(j for j in i[1:]) for i in list_points)
    return sum(list_of_points) / len(list_of_points)

def calculate_delta_point(list_points, avg_p, p):
    return list(sum(j - avg_p for j in i[:1]) for i in list_points) if p == 'x' else list(sum(j - avg_p for j in i[1:]) for i in list_points)

def calculate_slope_coefficient(delta_x, delta_y):
    numerator, denominator = 0, 0
    for x, y in zip(delta_x, delta_y):
        numerator += x * y
        denominator += x * x
    return numerator / denominator

def calculate_free_coefficient(x_avg, y_avg, slope_coeff):
    return y_avg - slope_coeff * x_avg

def calculate_matrix_P(delta_x, delta_y):
    P11 = sum(x * x for x in delta_x)
    P12 = sum(x * y for x, y in zip(delta_x, delta_y))
    P22 = sum(y * y for y in delta_y)
    discriminant = (P11 + P22) ** 2 - 4 * (P11 * P22 - P12 * P12)
    lambda1 = ((P11 + P22) + math.sqrt(discriminant)) / 2
    lambda2 = ((P11 + P22) - math.sqrt(discriminant)) / 2
    return min(lambda1, lambda2), max(lambda1, lambda2)


def calculate_coeffs(lambda_min, delta_x, delta_y):
    P11 = sum(x * x for x in delta_x)
    P12 = sum(x * y for x, y in zip(delta_x, delta_y))
    #P22 = sum(y * y for y in delta_y)

    #A, B = -P12 / (P11 - lambda_min), 1 if P11 - lambda_min != 0
    A, B = 0, 0
    if P11 - lambda_min != 0:
        A = -P12 / (P11 - lambda_min)
        B = 1
    elif P12 != 0:
        A = 1
        B = -(P11 - lambda_min) / P12
    elif P11 - lambda_min == 0 and P12 == 0:
        A = 1
        B = 0

    return A, B

def create_graph(list_of_points, slope_coeff, free_coefficient, coefficients):

    x_points = np.array(list(sum(j for j in i[:1]) for i in list_of_points))
    y_points = np.array(list(sum(j for j in i[1:]) for i in list_of_points))

    linear_regression_equation = slope_coeff * x_points + free_coefficient
    #rthogonal_regression_equation = coefficients[0] * x_points + coefficients[1] * y_points + (-coefficients[0] * x_points - coefficients[1] * y_points)

    #orthogonal_regression_equation = (-coefficients[0] * x_points - (-coefficients[0] * avg_x) - coefficients[1] * avg_y) / coefficients[1]

    plt.figure(figsize=(8, 6))
    plt.scatter(x_points, y_points, color="gray")
    plt.scatter(avg_x, avg_y, color="black")

    plt.plot(x_points, linear_regression_equation, color="blue", linewidth=2)     # blue - линейная регрессия
    #plt.plot(x_points, orthogonal_regression_equation, color="red", linewidth=2)

    #plt.xlabel("x", fontsize=12)
    #plt.ylabel("y", fontsize=12)
    #plt.title(f"График линейной функции y = {k}x + {b}", fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


if __name__ == '__main__':

    points = [[2, 4], [3, 1], [10, 23], [6, 1], [4, 4], [8, 2], [7, 9], [15, 9], [14, 1], [10, 3]]

    avg_x = calculate_average_x(points, 'x')
    print(f'x_ср = {avg_x}')

    avg_y = calculate_average_x(points, 'y')
    print(f'y_ср = {avg_y}')

    list_delta_x = calculate_delta_point(points, avg_x, 'x')
    print(f'Список дельта-x: {list_delta_x}')

    list_delta_y = calculate_delta_point(points, avg_y, 'y')
    print(f'Список дельта-y: {list_delta_y}')

    k = calculate_slope_coefficient(list_delta_x, list_delta_y)
    print(f'Угловой коэффициент k = {k}')

    b = calculate_free_coefficient(avg_x, avg_y, k)
    print(f'Свободный коэффициент b = {b}')

    min_lambda = calculate_matrix_P(list_delta_x, list_delta_y)[0]
    #max_lambda = calculate_matrix_P(list_delta_x, list_delta_y)[1]
    #print(min_lambda, max_lambda, calculate_matrix_P(list_delta_x, list_delta_y)[2])

    coeffs = calculate_coeffs(min_lambda, list_delta_x, list_delta_y)
    print(coeffs)

    create_graph(points, k, b, coeffs)
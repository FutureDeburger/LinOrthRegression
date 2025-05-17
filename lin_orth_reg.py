import matplotlib.pyplot as plt
import numpy as np
import math

def calculate_average_coordinate(list_points, p):
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

def calculate_lambdas(delta_x, delta_y):
    P11 = sum(x * x for x in delta_x)
    P12 = sum(x * y for x, y in zip(delta_x, delta_y))
    P22 = sum(y * y for y in delta_y)
    discriminant = (P11 + P22) ** 2 - 4 * (P11 * P22 - P12 * P12)
    lambda1 = ((P11 + P22) + math.sqrt(discriminant)) / 2
    lambda2 = ((P11 + P22) - math.sqrt(discriminant)) / 2
    return min(lambda1, lambda2), max(lambda1, lambda2)

def calculate_orth_coeffs(lambda_min, delta_x, delta_y, x_avg, y_avg):
    P11 = sum(x * x for x in delta_x)
    P12 = sum(x * y for x, y in zip(delta_x, delta_y))
    if P11 - lambda_min != 0 or P12 != 0:
        A = P12
        B = lambda_min - P11
    else:
        A = 1
        B = 0
    norm = math.sqrt(A ** 2 + B ** 2)
    A /= norm
    B /= norm
    C = -(A * x_avg + B * y_avg)
    return A, B, C

def create_graph(list_of_points, slope_coeff, free_coefficient, orth_coefficients):

    x_points = np.array(list(sum(j for j in i[:1]) for i in list_of_points))
    y_points = np.array(list(sum(j for j in i[1:]) for i in list_of_points))

    linear_regression_equation = slope_coeff * x_points + free_coefficient
    A, B, C = orth_coefficients
    orthogonal_regression_equation = (-A * x_points - C) / B if B != 0 else None

    plt.figure(figsize=(8, 6))
    plt.scatter(x_points, y_points, color="gray")
    plt.scatter(avg_x, avg_y, color="black")
    plt.plot(x_points, linear_regression_equation, color="blue", linewidth=2, label=f"Линейная регрессия")
    plt.plot(x_points, orthogonal_regression_equation, color="red", linewidth=2, label=f"Ортогональная регрессия") if B != 0 else plt.axvline(x=-C / A, color="red", linewidth=2, label="Ортогональная регрессия")

    plt.title(f"Линейная и ортогональная регрессия", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

if __name__ == '__main__':

    points = [[11, 4], [3, 1], [1, 2], [16, 1], [14, 4], [18, 12], [15, 9], [1, 9], [1, 4], [5, 3]]
    #points = [[1, 2], [2, 3], [3, 4], [4, 5], [2, 4], [3, 2], [4, 6], [5, 3]]
    #points = [[1, 10], [1, 12], [1, 8], [5, 5], [6, 6], [7, 7]]

    avg_x = calculate_average_coordinate(points, 'x')
    avg_y = calculate_average_coordinate(points, 'y')
    list_delta_x = calculate_delta_point(points, avg_x, 'x')
    list_delta_y = calculate_delta_point(points, avg_y, 'y')
    k = calculate_slope_coefficient(list_delta_x, list_delta_y)
    b = calculate_free_coefficient(avg_x, avg_y, k)
    min_lambda = calculate_lambdas(list_delta_x, list_delta_y)[0]
    coefficients_orth_line = calculate_orth_coeffs(min_lambda, list_delta_x, list_delta_y, avg_x, avg_y)
    create_graph(points, k, b, coefficients_orth_line)

    #print(f'Уравнение линейной регрессии: y = {k:.2}x + {b:.2}\nУравнение ортогональной регрессии: {coefficients_orth_line[0]:.2}x + {coefficients_orth_line[1]:.2}y + {coefficients_orth_line[2]:.2}')
    # print(f'x_ср = {avg_x}')
    # print(f'y_ср = {avg_y}')
    # print(f'Список дельта-x: {list_delta_x}')
    # print(f'Список дельта-y: {list_delta_y}')
    # print(f'Угловой коэффициент k = {k}')
    # print(f'Свободный коэффициент b = {b}')
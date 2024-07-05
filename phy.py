import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def naca_4digit_series(m, p, t, c=1, num_points=100):
    m /= 100
    p /= 10
    t /= 100
    x = np.linspace(0, c, num_points)
    yc = np.where( x < p * c, m / p*2 * (2 * p * x - x*2), m / (1 - p)2 * ((1 - 2 * p) + 2 * p * x - x*2))
    yt = 5 * t * ( 0.2969 * np.sqrt(x / c) - 0.126 * (x / c) - 0.3516 * (x / c)*2 + 0.2843 * (x / c)3 - 0.1015 * (x / c)*4)
    xu = x - yt * np.sin(np.arctan(yc))
    xl = x + yt * np.sin(np.arctan(yc))
    yu = yc + yt * np.cos(np.arctan(yc))
    yl = yc - yt * np.cos(np.arctan(yc))
    return xu, yu, xl, yl

def generate_airfoil_for_airplane(params):
    wing_span, max_camber, camber_location, thickness, aspect_ratio, chord_length = params

    x_upper, y_upper, x_lower, y_lower = naca_4digit_series(max_camber, camber_location, thickness, chord_length)
    return x_upper, y_upper, x_lower, y_lower

def calculate_material_density(max_mach):
    # Consider a function to determine material density based on maximum Mach number
    # Replace this placeholder with an appropriate calculation
    return 1.0  # Placeholder value

def generate_airfoil_coordinates(params, num_points=100):
    x_upper, y_upper, x_lower, y_lower = naca_4digit_series(params[3], params[0], params[1], params[2], num_points=num_points)

    # Combine upper and lower coordinates in the specified format
    coordinates = list(zip(x_upper[::-1], y_upper[::-1]))  # Reverse order for upper surface
    coordinates += list(zip(x_lower[1:], y_lower[1:]))  # Skip the first point for lower surface
    
    # Convert coordinates to strings in the required format
    coordinates_str = [f"{x:.4f} {y:.4f}" for x, y in coordinates]
    
    return coordinates_str


def plot_airfoil(x_upper, y_upper, x_lower, y_lower):
    plt.figure(figsize=(8, 6))
    plt.plot(x_upper, y_upper, label='Upper Surface')
    plt.plot(x_lower, y_lower, label='Lower Surface')
    plt.gca().set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Airfoil for Airplane')
    plt.legend()
    plt.grid()
    plt.show()

def lift_coefficient(alpha):
    # Lift coefficient equation (placeholder, replace with actual formula)
    return 2 * np.pi * alpha

def drag_coefficient(cl, cl_min, cd0, k):
    # Drag coefficient equation (placeholder, replace with actual formula)
    return cd0 + k * (cl - cl_min)**2

def velocity_factor(velocity):
    # Placeholder for velocity factor, you can replace this with a suitable formula
    return 1.0

def turbulence_factor():
    # Placeholder for turbulence factor, you can replace this with a suitable formula
    return 1.0

def objective_function(params, alpha):
    x_upper, y_upper, x_lower, y_lower = generate_airfoil_for_airplane(params)
    
    # Calculate lift and drag coefficients based on user input angle of attack
    cl = lift_coefficient(alpha)
    cl_min = 0.1  # Minimum lift coefficient (placeholder value)
    cd0 = 0.02  # Zero-lift drag coefficient (placeholder value)
    k = 0.05  # Coefficient for lift-induced drag (placeholder value)
    cd = drag_coefficient(cl, cl_min, cd0, k)
    
    # Placeholder for material density calculation (replace with actual function)
    material_density = calculate_material_density(0.0)
    
    # Placeholder for an objective metric (modify based on design goals)
    objective_metric = cl / cd * material_density * velocity_factor(params[4]) * turbulence_factor()
    
    return -objective_metric  # Negative as we aim to maximize the objective metric

# Get user inputs for airplane parameters
wing_span = float(input("Enter wing span (in meters): "))
max_camber = int(input("Enter maximum camber (0 to 9): "))
camber_location = int(input("Enter location of maximum camber (0 to 9): "))
thickness = int(input("Enter thickness (1 to 40): "))
aspect_ratio = float(input("Enter aspect ratio: "))
chord_length = float(input("Enter chord length (in meters): "))
alpha = float(input("Enter angle of attack (in degrees): "))

# Perform optimization to design the airfoil
initial_guess = [wing_span, max_camber, camber_location, thickness, aspect_ratio, chord_length]
bounds = [(10, 100), (0, 9), (0, 9), (1, 40), (1, 40), (0.5, 40)]

result = minimize(lambda params: objective_function(params, alpha), initial_guess, bounds=bounds)

# Extract optimized parameters
optimized_parameters = result.x
x_upper, y_upper, x_lower, y_lower = generate_airfoil_for_airplane(optimized_parameters)

# Plot the optimized airfoil
plot_airfoil(x_upper, y_upper, x_lower, y_lower)

# Save airfoil coordinates to a file
file_path = "optimized_airfoil_coordinates.txt"
coordinates = generate_airfoil_coordinates(optimized_parameters, num_points=100)

with open(file_path, 'w') as file:
    file.write("X Y\n")
    file.write("\n".join(coordinates))

print(f"Optimized airfoil coordinates saved to {file_path}")
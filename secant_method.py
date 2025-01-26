import numpy as np
import matplotlib.pyplot as plt

def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Secant Method to find the root of the equation f(x) = 0.
    
    Parameters:
    - f: Function for which the root is to be found.
    - x0, x1: Initial guesses.
    - tol: Tolerance for stopping criteria.
    - max_iter: Maximum number of iterations.
    
    Returns:
    - root: Approximated root of the function.
    - iterations: List of iterations for visualization.
    """
    iterations = [x0, x1]
    for _ in range(max_iter):
        if abs(f(x1) - f(x0)) < 1e-12:  # Avoid division by zero
            print("Division by zero encountered in secant method.")
            return None, iterations
        
        # Secant method formula
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        iterations.append(x2)
        
        if abs(x2 - x1) < tol:  # Check for convergence
            return x2, iterations
        
        x0, x1 = x1, x2  # Update points
    
    print("Secant method did not converge within the maximum number of iterations.")
    return None, iterations

# Example usage and visualization
def func(x):
    return x**3 - 6 * x**2 + 11 * x - 6  # Example function with roots at x=1, 2, 3

# Initial guesses
x0 = 2.5
x1 = 3.5

# Run the secant method
root, iter_points = secant_method(func, x0, x1)

# Plotting
x_vals = np.linspace(0, 4, 500)
y_vals = func(x_vals)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="f(x) = x^3 - 6x^2 + 11x - 6", color='blue')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

# Mark iterations
for i, xi in enumerate(iter_points[:-1]):
    plt.plot([xi, iter_points[i+1]], [func(xi), 0], 'ro--', label="Iteration" if i == 0 else "")

plt.scatter(root, 0, color='green', label=f"Root: {root:.6f}", zorder=5)
plt.title("Secant Method Visualization")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.show()

# K-Means Clustering Visualization

This project provides an interactive visualization of the K-Means clustering algorithm, implemented in Python using Matplotlib for plotting and Tkinter for the GUI. The visualization demonstrates the step-by-step process of K-Means clustering, showing how points are assigned to clusters and centroids are updated until convergence. All core calculations are performed using pure Python, with minimal NumPy usage confined to scipy.spatial.ConvexHull for drawing cluster boundaries.

The tool is designed for educational purposes, allowing users to explore K-Means clustering with customizable parameters, such as the number of points and clusters, and interactive controls for stepping through or animating the algorithm.

# Features
- Interactive GUI: Built with Tkinter, featuring controls for:
- Selecting the number of clusters (K=2 to 5).
- Specifying the number of points (with validation to ensure ≥ K).
- Resetting the simulation, stepping through iterations, or running/stopping the animation.
- Adjusting animation speed with "Faster" and "Slower" buttons (500ms to 3000ms delays).

# Step-by-Step Visualization:
- Points start black (except initial centroids) and are colored one by one during the assignment phase, reflecting their cluster assignment (e.g., red, green, blue).
- Displays distance lines, labels, and annotations during point assignments to illustrate the argmin process.
Cluster Boundaries: After convergence, draws convex hull boundaries around clusters (for clusters with ≥ 3 points) using scipy.spatial.ConvexHull, with transparent, cluster-colored polygons.
- Legend Placement: Legend is positioned below the x-axis for clarity, listing unassigned points, clusters, centroids, and boundaries (when converged).
- Convergence Detection: Automatically stops when centroids move less than 0.01 (squared distance), displaying the final iteration count.

# Robust Design:
- Uses centroid_indices to prevent IndexError during centroid handling.
- Handles edge cases for convex hulls (e.g., < 3 points or collinear points).
Validates user input for the number of points.
- Plot Area: Fixed 600x600 pixel area with adjusted bottom padding to accommodate the legend.

# Installation
# Prerequisites
- Python 3.6 or higher
- Required libraries:
    - matplotlib for plotting
    - scipy for convex hull calculations
    - tkinter for the GUI (usually included with Python; install python3-tk on Linux if needed)

# Install Dependencies
    - Install the required libraries using pip:


# Contributing
Contributions are welcome! Please:
- Fork the repository.
- Create a new branch (git checkout -b feature/your-feature).
- Commit your changes (git commit -m 'Add your feature').
- Push to the branch (git push origin feature/your-feature).
- Open a Pull Request.

# Acknowledgments
Inspired by the need to visualize K-Means clustering for educational purposes and concept building purposes.
Built with Matplotlib, SciPy, and Tkinter for a lightweight, interactive experience.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import ConvexHull
import tkinter as tk
from tkinter import ttk, messagebox

class KMeansVisualization:
    def __init__(self, root):
        # Store the Tkinter root window
        self.root = root
        # Set the window title
        self.root.title("K-Means Clustering Visualization")
        
        # Initialize clustering parameters
        self.k = 3  # Number of clusters
        self.num_points = 25  # Default number of points
        self.points = None  # Array to store data points
        self.centroids = None  # Array to store centroid positions
        self.assignments = None  # Array to store cluster assignments for each point
        self.centroid_indices = None  # Array to store indices of initial centroid points
        self.iteration = 0  # Current iteration count
        self.animating = False  # Flag to control animation state
        self.colors = ['red', 'green', 'blue', 'yellow', 'purple']  # Colors for clusters
        self.prev_centroids = None  # Store previous centroids for convergence check
        self.current_point_idx = 0  # Index of the point to visualize argmin for
        self.phase = 'assign'  # Track phase: 'assign' (show argmin) or 'update' (update centroids)
        self.delay = 1500  # Animation delay in milliseconds (default: 1500ms)
        self.delay_options = [500, 1000, 1500, 2000, 2500, 3000]  # Possible delay values
        self.after_id = None  # Store ID of scheduled after call to cancel if needed
        self.converged = False  # Flag to indicate convergence for drawing boundaries
        
        # Create and pack the control frame for GUI elements
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(pady=10)
        
        # Add label and combobox for cluster number selection
        tk.Label(self.control_frame, text="Number of Clusters (K):").pack(side=tk.LEFT)
        self.k_select = ttk.Combobox(self.control_frame, values=[2, 3, 4, 5], state="readonly")
        self.k_select.set(3)  # Set default value to 3
        self.k_select.pack(side=tk.LEFT, padx=5)
        self.k_select.bind("<<ComboboxSelected>>", self.update_k)
        
        # Add label and entry for number of points
        tk.Label(self.control_frame, text="Number of Points:").pack(side=tk.LEFT, padx=5)
        self.points_entry = tk.Entry(self.control_frame, width=5)
        self.points_entry.insert(0, str(self.num_points))  # Set default value
        self.points_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Apply Points", command=self.update_num_points).pack(side=tk.LEFT, padx=5)
        
        # Add buttons for simulation control
        tk.Button(self.control_frame, text="Reset Points", command=self.reset_simulation).pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Step", command=self.step).pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Run Animation", command=self.start_animation).pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Stop Animation", command=self.stop_animation).pack(side=tk.LEFT, padx=5)
        
        # Add buttons and label for animation speed control
        tk.Label(self.control_frame, text="Speed:").pack(side=tk.LEFT, padx=5)
        self.speed_label = tk.Label(self.control_frame, text=f"{self.delay} ms")
        self.speed_label.pack(side=tk.LEFT, padx=5)
        self.faster_button = tk.Button(self.control_frame, text="Faster", command=self.increase_speed)
        self.faster_button.pack(side=tk.LEFT, padx=5)
        self.slower_button = tk.Button(self.control_frame, text="Slower", command=self.decrease_speed)
        self.slower_button.pack(side=tk.LEFT, padx=5)
        # Update button states initially
        self.update_speed_buttons()
        
        # Create matplotlib figure and axis for plotting
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        # Adjust layout to accommodate legend below the plot
        self.fig.subplots_adjust(bottom=0.2)
        # Embed the plot in the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()
        
        # Initialize the simulation
        self.reset_simulation()
        
    def increase_speed(self):
        # Decrease the delay to increase animation speed
        current_idx = self.delay_options.index(self.delay)
        if current_idx > 0:
            self.delay = self.delay_options[current_idx - 1]
            self.speed_label.config(text=f"{self.delay} ms")
            self.update_speed_buttons()
            # If animating, cancel and restart with new delay
            if self.animating and self.after_id is not None:
                self.root.after_cancel(self.after_id)
                self.animate()
    
    def decrease_speed(self):
        # Increase the delay to decrease animation speed
        current_idx = self.delay_options.index(self.delay)
        if current_idx < len(self.delay_options) - 1:
            self.delay = self.delay_options[current_idx + 1]
            self.speed_label.config(text=f"{self.delay} ms")
            self.update_speed_buttons()
            # If animating, cancel and restart with new delay
            if self.animating and self.after_id is not None:
                self.root.after_cancel(self.after_id)
                self.animate()
    
    def update_speed_buttons(self):
        # Enable/disable speed buttons based on current delay
        current_idx = self.delay_options.index(self.delay)
        self.faster_button.config(state='normal' if current_idx > 0 else 'disabled')
        self.slower_button.config(state='normal' if current_idx < len(self.delay_options) - 1 else 'disabled')
    
    def initialize_points(self):
        # Generate user-specified number of random points in a 600x600 area
        self.points = np.random.rand(self.num_points, 2) * 500
        # Reset iteration count
        self.iteration = 0
        # Reset point index for argmin visualization
        self.current_point_idx = 0
        # Set phase to assignment
        self.phase = 'assign'
        # Initialize assignments to -1 (unassigned)
        self.assignments = np.full(len(self.points), -1, dtype=int)
        # Reset convergence flag
        self.converged = False
    
    def initialize_centroids(self):
        # Randomly select k points as initial centroids
        self.centroid_indices = np.random.choice(len(self.points), self.k, replace=False)
        self.centroids = self.points[self.centroid_indices].copy()
        # Initialize previous centroids for convergence check
        self.prev_centroids = np.zeros_like(self.centroids)
        # Assign initial centroids to their respective clusters
        for i, idx in enumerate(self.centroid_indices):
            self.assignments[idx] = i
    
    def update_assignments(self):
        # Reset assignments to -1 for non-centroid points
        for i in range(len(self.points)):
            if i not in self.centroid_indices:
                self.assignments[i] = -1
        # Assign each point to the nearest centroid
        for i in range(len(self.points)):
            if i not in self.centroid_indices:
                distances = np.sum((self.centroids - self.points[i]) ** 2, axis=1)
                self.assignments[i] = np.argmin(distances)
    
    def update_centroids(self):
        # Iterate through each cluster
        for i in range(self.k):
            # Get points assigned to current cluster
            cluster_points = self.points[self.assignments == i]
            # Update centroid if cluster is not empty
            if len(cluster_points) > 0:
                self.centroids[i] = np.mean(cluster_points, axis=0)
        # Update all assignments for the next iteration
        self.update_all_assignments()
    
    def check_convergence(self):
        # Check if centroids have moved significantly
        if self.prev_centroids is None:
            return False
        # Calculate maximum movement of any centroid
        max_movement = np.max(np.sum((self.centroids - self.prev_centroids) ** 2, axis=1))
        # Return True if movement is less than threshold (0.01)
        return max_movement < 0.01
    
    def update_all_assignments(self):
        # Update assignments for all points based on minimum distance
        for i in range(len(self.points)):
            if i not in self.centroid_indices:
                distances = np.sum((self.centroids - self.points[i]) ** 2, axis=1)
                self.assignments[i] = np.argmin(distances)
    
    def update_num_points(self):
        # Get and validate the number of points from the entry field
        try:
            num_points = int(self.points_entry.get())
            if num_points < self.k:
                messagebox.showerror("Invalid Input", f"Number of points must be at least {self.k} (number of clusters).")
                return
            if num_points <= 0:
                messagebox.showerror("Invalid Input", "Number of points must be positive.")
                return
            self.num_points = num_points
            # Reset the simulation with the new number of points
            self.reset_simulation()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer for the number of points.")
    
    def step(self):
        # Perform a single step based on the current phase
        if self.phase == 'assign':
            # In assignment phase, assign the current point (for visualization)
            if self.current_point_idx not in self.centroid_indices:
                # Calculate distances to all centroids
                distances = np.sum((self.centroids - self.points[self.current_point_idx]) ** 2, axis=1)
                # Assign point to cluster with minimum distance
                self.assignments[self.current_point_idx] = np.argmin(distances)
            # Move to next point
            self.current_point_idx += 1
            # If all points are processed, switch to update phase
            if self.current_point_idx >= len(self.points):
                self.phase = 'update'
                self.current_point_idx = 0
        elif self.phase == 'update':
            # In update phase, update centroids and check convergence
            self.prev_centroids = self.centroids.copy()
            self.update_centroids()
            self.iteration += 1
            self.phase = 'assign'
            # Check for convergence
            if self.check_convergence():
                self.animating = False
                self.converged = True
                self.update_plot()
                self.ax.set_title(f"Converged at Iteration: {self.iteration}")
                self.canvas.draw()
                return
        # Update the plot
        self.update_plot()
    
    def update_k(self, event):
        # Update number of clusters based on combobox selection
        self.k = int(self.k_select.get())
        # Check if number of points is sufficient
        if self.num_points < self.k:
            self.num_points = self.k
            self.points_entry.delete(0, tk.END)
            self.points_entry.insert(0, str(self.num_points))
        # Reset the simulation
        self.reset_simulation()
    
    def reset_simulation(self):
        # Stop any ongoing animation
        self.animating = False
        # Cancel any scheduled animation
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        # Initialize new points
        self.initialize_points()
        # Initialize new centroids
        self.initialize_centroids()
        # Update the plot
        self.update_plot()
    
    def start_animation(self):
        # Start the animation
        self.animating = True
        # Begin animation loop
        self.animate()
    
    def stop_animation(self):
        # Stop the animation
        self.animating = False
        # Cancel any scheduled animation
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
    
    def animate(self):
        # Continue animation if flag is True
        if self.animating:
            # Perform one step
            self.step()
            # Schedule next animation frame with current delay
            self.after_id = self.root.after(self.delay, self.animate)
    
    def update_plot(self):
        # Clear the current plot
        self.ax.clear()
        
        # If converged, draw convex hull boundaries for each cluster
        if self.converged:
            for i in range(self.k):
                # Get points assigned to current cluster
                cluster_points = self.points[self.assignments == i]
                # Only compute convex hull if cluster has at least 3 points
                if len(cluster_points) >= 3:
                    try:
                        # Compute convex hull
                        hull = ConvexHull(cluster_points)
                        # Get vertices of the hull
                        vertices = np.append(hull.vertices, hull.vertices[0])  # Close the polygon
                        # Plot the convex hull as a filled polygon
                        self.ax.fill(cluster_points[vertices, 0], cluster_points[vertices, 1],
                                    color=self.colors[i % len(self.colors)], alpha=0.2, label=f'Cluster {i} Boundary')
                    except:
                        # Skip if convex hull computation fails (e.g., collinear points)
                        pass
        
        # Plot unassigned points (black)
        unassigned_points = self.points[self.assignments == -1]
        self.ax.scatter(unassigned_points[:, 0], unassigned_points[:, 1], c='black', s=20, label='Unassigned')
        
        # Plot points for each cluster
        for i in range(self.k):
            cluster_points = self.points[self.assignments == i]
            self.ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=self.colors[i % len(self.colors)], s=20, label=f'Cluster {i}')
        
        # Plot centroids
        self.ax.scatter(self.centroids[:, 0], self.centroids[:, 1], c=self.colors[:self.k], s=100, marker='o', edgecolors='black', label='Centroids')
        
        # If in assignment phase, visualize argmin for the current point
        if self.phase == 'assign' and self.current_point_idx < len(self.points):
            # Highlight the current point
            point = self.points[self.current_point_idx]
            self.ax.scatter(point[0], point[1], c='black', s=50, marker='*', label='Selected Point')
            
            # Calculate distances to all centroids
            distances = np.sum((self.centroids - point) ** 2, axis=1)
            min_idx = np.argmin(distances)
            
            # Draw lines to all centroids with distance labels
            for j, centroid in enumerate(self.centroids):
                # Use solid line for minimum distance, dashed for others
                line_style = '-' if j == min_idx else '--'
                line_color = self.colors[j % len(self.colors)] if j == min_idx else 'grey'
                self.ax.plot([point[0], centroid[0]], [point[1], centroid[1]], c=line_color, alpha=0.5, linestyle=line_style)
                # Display distance
                dist = np.sqrt(distances[j])
                mid_x, mid_y = (point[0] + centroid[0]) / 2, (point[1] + centroid[1]) / 2
                self.ax.text(mid_x, mid_y, f'{dist:.1f}', fontsize=8, color=line_color)
            
            # Add annotation explaining the argmin process
            self.ax.text(6, 520, f'Point {self.current_point_idx} assigned to Cluster {min_idx}\n(Shortest distance: {np.sqrt(distances[min_idx]):.1f})',
                        bbox=dict(facecolor='white', alpha=0.8), fontsize=8)
        
        # Set plot limits
        self.ax.set_xlim(0, 500)
        self.ax.set_ylim(0, 500)
        # Set plot title with iteration and phase
        title = f"Iteration: {self.iteration}, {'Assigning Points' if self.phase == 'assign' else 'Updating Centroids'}"
        if self.phase == 'assign' and self.current_point_idx < len(self.points):
            title += f", Point {self.current_point_idx}"
        if self.converged:
            title = f"Converged at Iteration: {self.iteration}"
        self.ax.set_title(title)
        # Add legend below the x-axis
        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
        # Redraw the canvas
        self.canvas.draw()

if __name__ == "__main__":
    # Create Tkinter root window
    root = tk.Tk()
    # Initialize the application
    app = KMeansVisualization(root)
    # Start the Tkinter event loop
    root.mainloop()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Custom3DVisualizer:
    def __init__(self, data, x, y, z, colors=None, alpha=0.1, marker='o'):
        self.data = data
        self.x = x
        self.y = y
        self.z = z
        self.colors = colors
        self.alpha = alpha
        self.marker = marker

    def plot(self, ax, time_stretch=10.0):
        """Plot data with custom axis handling."""
        # Swap y and z for visualization purposes
        y_visual = self.y * time_stretch
        z_visual = self.z

        scatter = ax.scatter(self.x, z_visual, y_visual, c=self.colors, alpha=self.alpha, marker=self.marker)
        return scatter

def setup_plot():
    fig = plt.figure("Custom 3D Visualization", figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax

# Usage
if __name__ == '__main__':
    # main()
    fig, ax = setup_plot()
    visualizer = Custom3DVisualizer(data, x_values, y_values, z_values, colors='blue', alpha=0.5)
    visualizer.plot(ax, time_stretch=10.0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis (Vertical)')
    ax.set_zlabel('Y axis (Time/Depth stretched)')
    plt.show()

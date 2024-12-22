import numpy as np

class Body:

    def __init__(self, mass, position, velocity):
        self.mass = mass # mass of body
        self.position = np.array(position, dtype=np.float64) # position of body
        self.velocity = np.array(velocity, dtype=np.float64) # velocity of body
        self.force = np.array([0, 0], dtype=np.float64) # force on body

    # pos is position of other body
    # mass is mass of other body
    def calculate_force(self, pos, mass):

        G = 1 # gravitational constant
        soft = 0.35 # softening parameter

        # displacement vector from self to other_body
        r = pos - self.position 

        # force vector on self from other_body
        F = (G * mass * self.mass * r) / (((np.linalg.norm(r))**2 + soft**2)**(3/2))

        self.force = self.force + F # update force on body

# initialize bodies with random masses, positions, and velocities over set range
# num_bodies [scalar], mass_range [low, high], position_range [low, high], velocity_range [low, high]
def initialize_bodies(num_bodies, mass_range, position_range, velocity_range):

    masses = np.random.uniform(mass_range[0], mass_range[1], num_bodies)
    positions = np.random.uniform(position_range[0], position_range[1], (num_bodies, 2))
    velocities = np.random.uniform(velocity_range[0], velocity_range[1], (num_bodies, 2))

    # returning list of bodies
    return [Body(masses[i], positions[i], velocities[i]) for i in range(num_bodies)]

class Node:

    # bodies is list of bodies in simulation
    # points is list of indices of bodies in node
    # nodeposition is coordinates of node center
    # size is side length of node
    def __init__(self, bodies, points, nodeposition, size):

        # Offsets for child node positions
        self.quad_offsets = {
            1: np.array([-0.5, 0.5]),
            2: np.array([0.5, 0.5]),
            3: np.array([-0.5, -0.5]),
            4: np.array([0.5, -0.5])
        }

        self.points = points
        self.nodeposition = np.array(nodeposition)
        self.size = size
        self.Mass = 0
        self.CoM = None
        self.quad1 = None
        self.quad2 = None
        self.quad3 = None
        self.quad4 = None

        # Calculate mass and center of mass of node
        if len(self.points) > 0:
            # Calculate total mass of node
            self.Mass = sum(bodies[i].mass for i in self.points)
            # Calculate center of mass of node
            self.CoM = sum(bodies[i].mass * bodies[i].position for i in self.points) / self.Mass
        
    # bodies is list of all bodies in simulation
    # nodelist is list of all nodes in quadtree
    # simsize is side length of square containing entire simulation
    # subdivide node into quadrants
    def subdivide(self, bodies, nodelist):

        # Initialize quadrant body index lists
        quad1points = []
        quad2points = []
        quad3points = []
        quad4points = []

        # Sort bodies in node into appropriate quadrants
        for i in self.points:
            
            if  bodies[i].position[0] <= self.nodeposition[0]:
                if bodies[i].position[1] >= self.nodeposition[1]:
                    quad1points.append(i)
                else:
                    quad3points.append(i)
            else:
                if bodies[i].position[1] >= self.nodeposition[1]:
                    quad2points.append(i)
                else:
                    quad4points.append(i)
                    
        # Quad points => [1] [2]
        #             => [3] [4]
        # create child nodes and append to nodelist
        nodelist.append(Node(bodies, quad1points, self.nodeposition + (self.size / 2) * self.quad_offsets[1], self.size / 2))
        self.quad1 = len(nodelist) - 1
        nodelist.append(Node(bodies, quad2points, self.nodeposition + (self.size / 2) * self.quad_offsets[2], self.size / 2))
        self.quad2 = len(nodelist) - 1
        nodelist.append(Node(bodies, quad3points, self.nodeposition + (self.size / 2) * self.quad_offsets[3], self.size / 2))
        self.quad3 = len(nodelist) - 1
        nodelist.append(Node(bodies, quad4points, self.nodeposition + (self.size / 2) * self.quad_offsets[4], self.size / 2))
        self.quad4 = len(nodelist) - 1

    # bodies is list of all bodies in simulation
    # nodelist is list of all nodes in quadtree
    # simsize is side length of square containing entire simulation
    # check if current node contains more than one body and call subdivide if it does
    def checkQuad(self, bodies, nodelist):

        # Prevent small nodes to avoid recursion limit
        if self.size < 1:
            return

        # Avoid subdividing nodes with 1 body
        if len(self.points) <= 1 :
            return
        
        # Subdivide nodes with more than 1 body
        if len(self.points) > 1:

            # Subdivide node
            self.subdivide(bodies, nodelist)

            # Calls itself recursively
            if len(nodelist[self.quad1].points) > 0:
                nodelist[self.quad1].checkQuad(bodies, nodelist)
            if len(nodelist[self.quad2].points) > 0:
                nodelist[self.quad2].checkQuad(bodies, nodelist)
            if len(nodelist[self.quad3].points) > 0:
                nodelist[self.quad3].checkQuad(bodies, nodelist)
            if len(nodelist[self.quad4].points) > 0:
                nodelist[self.quad4].checkQuad(bodies, nodelist)

class Quadtree:
    
# Takes bodies of simulation, and simulation size and generates a quadtree with a list of nodes
# Each node contains the nessecary information in a node, as well as the index numbers of it's children
# bodies is all bodies within the simulation in the bodies data structure
    def __init__(self, bodies, simsize):
        self.bodies = bodies
        self.simsize = simsize
        self.nodelist = [Node(bodies, list(range(len(bodies))), [0, 0], simsize)]
        
        self.nodelist[0].checkQuad(self.bodies, self.nodelist)

    # determine if ratio of node size to distance satisfies threshold for approximation
    def check_distance(self, node_index, body_index):

        # Simulation accuracy parameter
        theta = 0.5

        if self.nodelist[node_index].CoM is None:
            return False
        
        # Distance from body to center of mass of node
        distance = np.linalg.norm(self.bodies[body_index].position - self.nodelist[node_index].CoM)

        if self.nodelist[node_index].size / distance < theta:
            return True
        
        else:
            return False
        
    # determine if the current node contains the body we are calculating forces on
    def contains_self(self, node_index, body_index):

        if body_index in self.nodelist[node_index].points:
            return True
        
        else:
            return False

    # Traverse the quadtree and calculate forces
    def traverse_quadtree(self, node_index, body_index):
        
        # Check if body is in current node
        if self.contains_self(node_index, body_index):

            # Check if current node has children
            if self.nodelist[node_index].quad1 is not None:

                self.traverse_quadtree(self.nodelist[node_index].quad1, body_index)
                self.traverse_quadtree(self.nodelist[node_index].quad2, body_index)
                self.traverse_quadtree(self.nodelist[node_index].quad3, body_index)
                self.traverse_quadtree(self.nodelist[node_index].quad4, body_index)

            # Current node has no children
            else:
                # Calculate force on bodies in leaf node containing self
                for point in self.nodelist[node_index].points:
                    if point != body_index:
                        self.bodies[body_index].calculate_force(self.bodies[point].position, 
                                                                self.bodies[point].mass)
            
        # Body not in current node
        else:
            if self.check_distance(node_index, body_index):
                # Calculate force on group here
                self.bodies[body_index].calculate_force(self.nodelist[node_index].CoM, self.nodelist[node_index].Mass)

            # Current node fails distance check
            else:
                if self.nodelist[node_index].quad1 is not None:

                    self.traverse_quadtree(self.nodelist[node_index].quad1, body_index)
                    self.traverse_quadtree(self.nodelist[node_index].quad2, body_index)
                    self.traverse_quadtree(self.nodelist[node_index].quad3, body_index)
                    self.traverse_quadtree(self.nodelist[node_index].quad4, body_index)

                # Current node has no children
                else:
                    if len(self.nodelist[node_index].points) == 0:
                        # Current node is empty
                        return
                    else:     
                        # Calculate force on bodies in leaf node here
                        for point in self.nodelist[node_index].points:
                            self.bodies[body_index].calculate_force(self.bodies[point].position, self.bodies[point].mass)
                        
class Simulation:

    def __init__(self, dt, num_frames, simsize):
        
        self.dt = dt
        self.num_frames = num_frames
        self.simsize = simsize
        
    def generate_bodies(self, num_bodies, mass_range, position_range, velocity_range):

        self.bodies = initialize_bodies(num_bodies, mass_range, position_range, velocity_range)

    def generate_solar_system(self):
        # solar system
        self.bodies = [
            Body(mass = 1, position=[0, 0], velocity=[0, 0]), # sun
            Body(mass = 3.00e-6, position=[1, 0], velocity=[0, 1]), # earth
            Body(mass = 3.21e-7, position=[1.5, 0], velocity=[0, np.sqrt(1/1.5)]), # mars
            Body(mass = 9.55e-4, position=[5.2, 0], velocity=[0, np.sqrt(1/5.2)]), # jupiter
            Body(mass = 2.86e-4, position=[9.5, 0], velocity=[0, np.sqrt(1/9.5)]), # saturn
            Body(mass = 4.36e-5, position=[19.2, 0], velocity=[0, np.sqrt(1/19.2)]), # uranus
            Body(mass = 5.15e-5, position=[30.1, 0], velocity=[0, np.sqrt(1/30.1)]) # neptune
        ]

    # function to run simulation and store output in matrix
    def run_sim(self):
        
        # Simulation time steps
        time_steps = np.arange(0, self.num_frames * self.dt, self.dt)

        # Initialize matrix to store position data
        self.data = np.zeros((len(time_steps), len(self.bodies), 2))

        # List to store node positions and sizes for each frame
        self.node_positions = []
        self.node_sizes = []

        for time_index, time in enumerate(time_steps):

            # Generate quadtree
            quadtree = Quadtree(self.bodies, self.simsize)

            # Store the node positions and sizes for this frame
            node_positions = np.array([node.nodeposition for node in quadtree.nodelist])
            self.node_positions.append(node_positions)
            node_sizes = np.array([node.size for node in quadtree.nodelist])
            self.node_sizes.append(node_sizes)

            # Leapfrog integration
            for n, body in enumerate(self.bodies):
                body.force = np.array([0, 0]) # Reset force on body
                quadtree.traverse_quadtree(0, n) # Calculate forces
                body.velocity += 0.5 * body.force / body.mass * self.dt # Update velocity (half step)
                body.position += body.velocity * self.dt # Update position (full step)
                self.data[time_index, n] = [body.position[0], body.position[1]] # Store position data

            quadtree = Quadtree(self.bodies, self.simsize)

            for n, body in enumerate(self.bodies):
                body.force = np.array([0, 0]) # Reset force on body
                quadtree.traverse_quadtree(0, n) # Recalculate force
                body.velocity += 0.5 * body.force / body.mass * self.dt # Update velocity (second half step)

    def run_sim_euler(self):

        # Simulation time steps
        time_steps = np.arange(0, self.num_frames * self.dt, self.dt)

        # Initialize matrix to store position data
        self.data = np.zeros((len(time_steps), len(self.bodies), 2))

        # List to store node positions and sizes for each frame
        self.node_positions = []
        self.node_sizes = []

        for time_index, time in enumerate(time_steps):

            # generate quadtree
            quadtree = Quadtree(self.bodies, self.simsize)

            # Store the node positions and sizes for this frame
            node_positions = np.array([node.nodeposition for node in quadtree.nodelist])
            self.node_positions.append(node_positions)
            node_sizes = np.array([node.size for node in quadtree.nodelist])
            self.node_sizes.append(node_sizes)

            # Euler integration
            for n, body in enumerate(self.bodies):
                body.force = np.array([0, 0]) # Reset force on body
                quadtree.traverse_quadtree(0, n) # Calculate forces
                body.velocity += body.force / body.mass * self.dt # Update velocity
                body.position += body.velocity * self.dt # Update position (full step)
                self.data[time_index, n] = [body.position[0], body.position[1]] # Store position data
            
    # function to generate video from output matrix
    def save_animation(self, filename = 'simulation.mp4'):

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.patches import Rectangle
        from matplotlib import colormaps

        fps = 30

        num_frames = self.data.shape[0]
        num_bodies = self.data.shape[1]

        # Set up the figure
        fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi = 100)
        ax.set_xlim(self.simsize * -0.5, self.simsize * 0.5)
        ax.set_ylim(self.simsize * -0.5, self.simsize * 0.5)
        ax.set_title(f'{len(self.bodies)} Body Barnes-Hut Simulation')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_facecolor("black")  # Set background color
        fig.patch.set_facecolor("lightgray")  # Outside the plot

        # Sizes
        sizes = [50*body.mass**(1/3) for body in self.bodies]

        # Initialize points
        points = ax.scatter([0] * num_bodies, [0] * num_bodies, s=sizes, c='yellow')

        # Scatter plot for quadtree node centers
        #node_points = ax.scatter([], [], s=20, c='cyan', label="Quadtree Nodes")

        # Initialize list of node boundaries
        rectangles = []

        # Update function for animation
        def update(frame):

            # Clear previous rectangles
            for rect in rectangles:
                rect.remove()
            rectangles.clear()

            # Update body positions
            positions = self.data[frame]
            points.set_offsets(positions)

            # Update node positions
            node_positions = self.node_positions[frame]
            #node_points.set_offsets(node_positions)

            # Add rectangles for each node
            node_sizes = self.node_sizes[frame]
            for pos, size in zip(node_positions, node_sizes):
                x, y = pos
                rect = Rectangle(
                    (x - size / 2, y - size / 2),  # Bottom-left corner
                    size, size,  # Width and height
                    linewidth=1,
                    edgecolor='white',
                    facecolor='none'
                )
                ax.add_patch(rect)
                rectangles.append(rect)

            # return points, node_points, *rectangles
            return points, *rectangles

        # Create the animation
        ani = FuncAnimation(fig, update, frames=num_frames)

        # Save the animation
        ani.save(filename, fps=fps, writer="ffmpeg")
        print(f"Animation saved to {filename}")

        plt.close(fig)  # Close the figure to free resources

        return
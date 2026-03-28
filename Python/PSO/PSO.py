import numpy as np
from datetime import datetime
from typing import Tuple, List
import time

def sensors_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Initialize sensor positions and TDOA measurement data."""
    # Speed of EM Wave
    speed_of_EM = 299792458

    # Sensor placement
    receiver1 = np.array([1, 3, 1.4])
    receiver2 = np.array([1, 5, 1.8])
    receiver3 = np.array([1, 7, 1.4])
    receiver4 = np.array([1, 9, 1.8])

    # Data predict format: X, Y, Z, T21, T31, T41
    # Time unit is in ns
    data_predict = np.array([
        # location 2
        # [1.999993 + 1, -0.0122108 + 1, 1.2, 5.3838757, 11.3425882, 17.8626851]
        [2.999993, 0.9877892, 1.2, 5.3838757, 11.3425882, 17.8626851],
        # Add more data points as needed
    ])

    return receiver1, receiver2, receiver3, receiver4, speed_of_EM, data_predict

def sorting_data_predict(data_predict: np.ndarray, data_set: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract true source location and TDOA measurements for a specific dataset."""
    true_source_location = data_predict[data_set-1, :3]
    TDOA_N = (data_predict[data_set-1, 3:])*1e-9  # Convert to seconds
    return true_source_location, TDOA_N

def objective_function(position: np.ndarray, receivers: List[np.ndarray], speed_of_EM: float, TDOA_N: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    """Calculate objective function value for PSO."""
    x0, y0, z0 = position
    
    # Calculate distances to each receiver
    d1 = np.sqrt(np.sum((position - receivers[0])**2))
    d2 = np.sqrt(np.sum((position - receivers[1])**2))
    d3 = np.sqrt(np.sum((position - receivers[2])**2))
    d4 = np.sqrt(np.sum((position - receivers[3])**2))

    # Calculate residuals
    r1 = d2 - d1 - (speed_of_EM * TDOA_N[0])
    r2 = d3 - d1 - (speed_of_EM * TDOA_N[1])
    r3 = d4 - d1 - (speed_of_EM * TDOA_N[2])
    
    return r1, r2, r3, d1, d2, d3, d4

def compute_error(true_source_location: np.ndarray, estimated_location: np.ndarray, receivers: List[np.ndarray]) -> Tuple[float, float]:
    """Compute error metrics between true and estimated locations."""
    euclidean_distance = np.sqrt(np.sum((true_source_location - estimated_location)**2))
    # Calculate percentage error (assuming maximum possible error is based on domain size)
    sum_total = np.linalg.norm((np.abs(estimated_location - receivers[0]) + np.abs(estimated_location - receivers[1]) + np.abs(estimated_location - receivers[2]) + np.abs(estimated_location - receivers[3]))/4)
    percentage_error = (euclidean_distance / sum_total) * 100
    return euclidean_distance, percentage_error

def pso_algorithm():
    """Main PSO algorithm implementation."""
    # Get sensor data
    receiver1, receiver2, receiver3, receiver4, speed_of_EM, data_predict = sensors_data()
    receivers = [receiver1, receiver2, receiver3, receiver4]
    
    num_of_loop = len(data_predict)
    results = []

    for data_set in range(1, num_of_loop + 1):
        start_time = datetime.now()

        # Get true source location and TDOA measurements
        true_source_location, TDOA_N = sorting_data_predict(data_predict, data_set)

        # PSO parameters
        max_iterations_pso = 50000
        convergence_threshold = 1e-12
        convergence_iter_threshold = 1000
        num_particles = 1000
        inertia_weight = 0.1
        c1 = 3.0  # Cognitive coefficient
        c2 = 1.0  # Social coefficient

        # Initialize particles within boundary
        particles_position = np.random.uniform(
            low=[0.5, 0.5, 0.1],
            high=[11, 11, 2],
            size=(num_particles, 3)
        )
        particles_velocity = np.random.rand(num_particles, 3)
        
        # Initialize best positions and values
        personal_best_positions = particles_position.copy()
        personal_best_values = np.ones(num_particles) * np.inf
        global_best_value = np.inf
        global_best_position = particles_position[0].copy()
        
        convergence_counter = 0
        prev_global_best_value = 0

        # Main PSO loop
        for iter_pso in range(max_iterations_pso):
            for i in range(num_particles):
                # Calculate objective function
                r1, r2, r3, *_ = objective_function(
                    particles_position[i],
                    receivers,
                    speed_of_EM,
                    TDOA_N
                )
                
                # Calculate RMSE
                objective_value = np.sqrt((r1**2 + r2**2 + r3**2)/3)
                
                # Update personal best
                if objective_value < personal_best_values[i]:
                    personal_best_values[i] = objective_value
                    personal_best_positions[i] = particles_position[i].copy()
                
                # Update global best
                if objective_value < global_best_value:
                    global_best_value = objective_value
                    global_best_position = particles_position[i].copy()
            
            # Update velocities and positions
            r1, r2 = np.random.rand(num_particles, 1), np.random.rand(num_particles, 1)
            particles_velocity = (inertia_weight * particles_velocity +
                               c1 * r1 * (personal_best_positions - particles_position) +
                               c2 * r2 * (global_best_position - particles_position))
            
            particles_position += particles_velocity
            
            # Apply boundary conditions
            particles_position = np.clip(particles_position, [0.5, 0.5, 0.1], [11, 11, 2])
            
            # Check convergence
            if abs(global_best_value - prev_global_best_value) < convergence_threshold:
                convergence_counter += 1
            else:
                convergence_counter = 0
            
            if convergence_counter >= convergence_iter_threshold:
                print(f'Convergence reached after {iter_pso} iterations.')
                break
            
            prev_global_best_value = global_best_value
        
        # Compute final errors
        euclidean_distance, percentage_error = compute_error(true_source_location, global_best_position, receivers)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Store results
        results.append({
            'data_set': data_set,
            'percentage_error': round(percentage_error, 2),
            'euclidean_distance': round(euclidean_distance, 3),
            'computation_time': round(duration, 2),
            'estimated_location': [round(x, 2) for x in global_best_position],
            'true_location': [round(x, 2) for x in true_source_location]
        })
        
        # Display results
        print(f"\nResults for dataset {data_set}:")
        print(f"Estimated location: {results[-1]['estimated_location']}")
        print(f"True location: {results[-1]['true_location']}")
        print(f"Euclidean distance error: {results[-1]['euclidean_distance']}")
        print(f"Percentage error: {results[-1]['percentage_error']}%")
        print(f"Computation time: {results[-1]['computation_time']} seconds")

    return results

if __name__ == "__main__":
    results = pso_algorithm()
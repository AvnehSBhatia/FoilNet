import torch
import numpy as np
import matplotlib.pyplot as plt
from naca_trainer import AF512toXYNet, predict_xy_coordinates, naca_to_af512, naca4_digit_to_coordinates
import random
import copy
from typing import List, Tuple, Optional
import json
import os

try:
    import neuralfoil
    print("NeuralFoil imported successfully")
except ImportError:
    print("NeuralFoil not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "neuralfoil"])
    import neuralfoil
    print("Neuralfoil installed and imported")

class GeneticAF512Optimizer:
    
    def __init__(self, 
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 generations: int = 100,
                 wingspan: float = 6.0,
                 chord: float = 0.8,
                 airspeed: float = 30.0,
                 alpha_range: Tuple[float, float] = (-5, 15),
                 num_points: int = 512,
                 target_lift: float = None,
                 image_callback: callable = None,
                 early_stopping_patience: int = 10,
                 early_stopping_tolerance: float = 0.001):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.wingspan = wingspan
        self.chord = chord
        self.airspeed = airspeed
        self.alpha_range = alpha_range
        self.num_points = num_points
        self.target_lift = target_lift
        
        airspeed_fts = airspeed * 1.467
        density = 0.002377
        dynamic_viscosity = 3.74e-7
        self.reynolds_number = (density * airspeed_fts * chord) / dynamic_viscosity
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AF512toXYNet(input_size=1024, output_size=2048, hidden_sizes=[512, 256, 128])
        
        try:
            self.model.load_state_dict(torch.load('af512_to_xy_model.pth', map_location=self.device))
            self.model.eval()
            print("AF512 model loaded successfully")
        except Exception as e:
            print(f"Error loading AF512 model: {e}")
            raise
        
        self.population = []
        self.fitness_history = []
        self.best_individual_history = []
        self.image_callback = image_callback
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_tolerance = early_stopping_tolerance
        
        self.current_noise_std = 0.02
        self.min_noise_std = 0.005
        self.noise_decrease_factor = 0.99
        self.stagnation_threshold = 25
        self.fitness_history = []
        
        self.adaptive_mode = False
        self.adaptive_threshold = 25
        self.peak_ld_tolerance = 0.03
        self.alpha_range = np.linspace(-5, 15, 21)
        
    def create_individual(self) -> dict:
        x_points = np.linspace(0, 1, self.num_points)
        
        thickness = np.random.uniform(0.08, 0.16)
        thickness_pos = np.random.uniform(0.2, 0.8)
        
        a = 0.3 + 0.4 * (1 - abs(thickness_pos - 0.5) / 0.5)
        b = thickness / 2
        
        thickness_dist = np.zeros(self.num_points)
        for i, x in enumerate(x_points):
            dx = (x - thickness_pos) / a
            if abs(dx) <= 1:
                thickness_dist[i] = b * np.sqrt(1 - dx**2)
            else:
                thickness_dist[i] = 0
        
        upper_dist = thickness_dist
        lower_dist = -thickness_dist
        
        upper_dist = np.clip(upper_dist, 0.001, 0.16)
        lower_dist = np.clip(lower_dist, -0.16, -0.001)
        
        upper_dist[0] = 0.0
        lower_dist[0] = 0.0
        
        leading_edge_thickness = upper_dist[-1] - lower_dist[-1]
        if leading_edge_thickness < 0.01:
            center = (upper_dist[-1] + lower_dist[-1]) / 2
            upper_dist[-1] = center + 0.005
            lower_dist[-1] = center - 0.005
        
        af512_data = np.column_stack([upper_dist, lower_dist])
        
        return {
            'af512_data': af512_data,
            'fitness': None,
            'xy_coordinates': None,
            'aerodynamic_data': None
        }
    
    def create_individual_from_naca(self, naca_code: str) -> dict:
        try:
            af512_data = naca_to_af512(naca_code, self.num_points)
            return {
                'af512_data': af512_data,
                'fitness': None,
                'xy_coordinates': None,
                'aerodynamic_data': None
            }
        except Exception as e:
            print(f"Error creating from NACA {naca_code}: {e}")
            return self.create_individual()
    
    def initialize_population(self, initial_naca: str = None):
        print(f"Initializing population of {self.population_size} individuals...")
        self.population = [self.create_individual() for _ in range(self.population_size)]
        
        if initial_naca:
            print(f"Using initial NACA: {initial_naca}")
            self.population[0] = self.create_individual_from_naca(initial_naca)
    
    def af512_to_xy_coordinates(self, af512_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            pred_x, pred_y = predict_xy_coordinates(self.model, af512_data, self.device)
            
            pred_x = np.clip(pred_x, 0, 1)
            
            return pred_x, pred_y
            
        except Exception as e:
            print(f"Error converting AF512 to coordinates: {e}")
            x = np.linspace(0, 1, self.num_points)
            y = np.zeros_like(x)
            return x, y
    
    def evaluate_aerodynamics(self, x_coords: np.ndarray, y_coords: np.ndarray) -> dict:
        try:
            coords = np.column_stack([x_coords, y_coords])
            
            if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                raise ValueError("Invalid coordinates detected")
            
            if np.any(x_coords < 0) or np.any(x_coords > 1):
                raise ValueError("X coordinates out of range [0,1]")
            
            best_ld = 0.0
            best_alpha = 0.0
            best_cl = 0.0
            best_cd = 1.0
            ld_values = []
            
            try:
                for alpha in self.alpha_range:
                    results = neuralfoil.get_aero_from_coordinates(
                        coordinates=coords,
                        alpha=alpha,
                        Re=self.reynolds_number,
                        model_size="xxxlarge"
                    )
                    
                    cl = float(results['CL'].item() if hasattr(results['CL'], 'item') else results['CL'])
                    cd = float(results['CD'].item() if hasattr(results['CD'], 'item') else results['CD'])
                    ld_ratio = cl / cd if cd > 0 else 0
                    ld_values.append(ld_ratio)
                    
                    if ld_ratio > best_ld:
                        best_ld = ld_ratio
                        best_alpha = alpha
                        best_cl = cl
                        best_cd = cd
                
                area_under_curve = np.trapz(ld_values, self.alpha_range)
                
                return {
                    'CL': best_cl,
                    'CD': best_cd,
                    'L/D': best_ld,
                    'alpha': best_alpha,
                    'peak_alpha': best_alpha,
                    'area_under_curve': area_under_curve,
                    'ld_values': ld_values
                }
                
            except (RuntimeWarning, RuntimeError) as e:
                print(f"NeuralFoil runtime error: {e}")
                return {
                    'CL': 0.0,
                    'CD': 1.0,
                    'L/D': 0.0,
                    'alpha': 0.0,
                    'peak_alpha': 0.0,
                    'area_under_curve': 0.0,
                    'ld_values': [0.0] * len(self.alpha_range)
                }
            
        except Exception as e:
            print(f"Error in aerodynamic evaluation: {e}")
            return {
                'CL': 0.0,
                'CD': 1.0,
                'L/D': 0.0,
                'alpha': 0.0,
                'peak_alpha': 0.0,
                'area_under_curve': 0.0,
                'ld_values': [0.0] * len(self.alpha_range)
            }
    
    def evaluate_fitness(self, individual: dict) -> float:
        try:
            x_coords, y_coords = self.af512_to_xy_coordinates(individual['af512_data'])
            individual['xy_coordinates'] = (x_coords, y_coords)
            
            aero_data = self.evaluate_aerodynamics(x_coords, y_coords)
            individual['aerodynamic_data'] = aero_data
            
            if self.target_lift is not None:
                if aero_data['CL'] >= self.target_lift:
                    fitness = self._calculate_adaptive_fitness(aero_data)
                else:
                    lift_penalty = (self.target_lift - aero_data['CL']) * 1000
                    fitness = -lift_penalty
            else:
                fitness = self._calculate_adaptive_fitness(aero_data)
            
            individual['fitness'] = fitness
            return fitness
            
        except Exception as e:
            print(f"Error evaluating fitness: {e}")
            individual['fitness'] = 0.0
            return 0.0
    
    def _calculate_adaptive_fitness(self, aero_data: dict) -> float:
        if not self.adaptive_mode:
            return aero_data['L/D']
        else:
            if not self.population:
                return aero_data['L/D']
            
            best_peak_ld = max(ind['aerodynamic_data']['L/D'] for ind in self.population 
                             if ind['aerodynamic_data'] is not None)
            
            if aero_data['L/D'] >= best_peak_ld * (1 - self.peak_ld_tolerance):
                normalized_area = aero_data['area_under_curve'] / 20.0
                return aero_data['L/D'] + normalized_area * 0.1
            else:
                return aero_data['L/D']
    
    def evaluate_population(self):
        print("Evaluating population fitness...")
        for i, individual in enumerate(self.population):
            if i % 10 == 0:
                print(f"   Evaluating individual {i+1}/{len(self.population)}")
            self.evaluate_fitness(individual)
    
    def select_parents(self) -> Tuple[dict, dict]:
        tournament_size = 3
        
        def tournament_select():
            tournament = random.sample(self.population, tournament_size)
            return max(tournament, key=lambda x: x['fitness'])
        
        parent1 = tournament_select()
        parent2 = tournament_select()
        
        return parent1, parent2
    
    def crossover(self, parent1: dict, parent2: dict) -> Tuple[dict, dict]:
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        af512_1 = child1['af512_data']
        af512_2 = child2['af512_data']
        
        blend_factor = random.uniform(0.3, 0.7)
        
        af512_1 = af512_1 * blend_factor + af512_2 * (1 - blend_factor)
        af512_2 = af512_2 * blend_factor + af512_1 * (1 - blend_factor)
        
        from scipy.ndimage import gaussian_filter1d
        af512_1[:, 0] = gaussian_filter1d(af512_1[:, 0], sigma=2)
        af512_1[:, 1] = gaussian_filter1d(af512_1[:, 1], sigma=2)
        af512_2[:, 0] = gaussian_filter1d(af512_2[:, 0], sigma=2)
        af512_2[:, 1] = gaussian_filter1d(af512_2[:, 1], sigma=2)
        
        for af512_data in [af512_1, af512_2]:
            af512_data[:, 0] = np.clip(af512_data[:, 0], 0.001, 0.16)
            af512_data[:, 1] = np.clip(af512_data[:, 1], -0.16, -0.001)
            af512_data[0, :] = 0.0
        
        child1['fitness'] = None
        child2['fitness'] = None
        
        return child1, child2
    
    def mutate(self, individual: dict):
        if random.random() > self.mutation_rate:
            return
        
        af512_data = individual['af512_data']
        
        num_points_to_mutate = random.randint(1, 5)
        mutation_strength = 0.01
        
        for _ in range(num_points_to_mutate):
            point_idx = random.randint(0, self.num_points - 1)
            
            noise_upper = np.random.normal(0, mutation_strength)
            noise_lower = np.random.normal(0, mutation_strength)
            
            af512_data[point_idx, 0] += noise_upper
            af512_data[point_idx, 1] += noise_lower
        
        af512_data[:, 0] = np.clip(af512_data[:, 0], 0.001, 0.16)
        af512_data[:, 1] = np.clip(af512_data[:, 1], -0.16, -0.001)
        
        af512_data[0, :] = 0.0
        
        from scipy.ndimage import gaussian_filter1d
        af512_data[:, 0] = gaussian_filter1d(af512_data[:, 0], sigma=3)
        af512_data[:, 1] = gaussian_filter1d(af512_data[:, 1], sigma=3)
        
        leading_edge_thickness = af512_data[-1, 0] - af512_data[-1, 1]
        if leading_edge_thickness < 0.01:
            center = (af512_data[-1, 0] + af512_data[-1, 1]) / 2
            af512_data[-1, 0] = center + 0.005
            af512_data[-1, 1] = center - 0.005
        
        individual['fitness'] = None
    
    def add_noise_to_individual(self, individual, noise_std=0.01):
        if 'af512_data' in individual:
            noise = np.random.normal(0, noise_std, individual['af512_data'].shape)
            individual['af512_data'] += noise
            
            from scipy.ndimage import gaussian_filter1d
            individual['af512_data'][:, 0] = gaussian_filter1d(individual['af512_data'][:, 0], sigma=3)
            individual['af512_data'][:, 1] = gaussian_filter1d(individual['af512_data'][:, 1], sigma=3)
            
            individual['af512_data'][:, 0] = np.clip(individual['af512_data'][:, 0], 0.001, 0.16)
            individual['af512_data'][:, 1] = np.clip(individual['af512_data'][:, 1], -0.16, -0.001)
            individual['af512_data'][0, :] = 0.0
            
            individual['fitness'] = None
            individual['xy_coordinates'] = None
            individual['aerodynamic_data'] = None
    
    def add_noise_to_population(self, noise_std=None):
        if noise_std is None:
            noise_std = self.current_noise_std
        print(f"   Adding noise (std={noise_std:.3f}) to population...")
        for individual in self.population:
            self.add_noise_to_individual(individual, noise_std)
    
    def adjust_noise_dynamically(self):
        if len(self.fitness_history) < self.stagnation_threshold:
            return
        
        recent_fitnesses = [h['best_fitness'] for h in self.fitness_history[-self.stagnation_threshold:]]
        if len(recent_fitnesses) >= self.stagnation_threshold:
            best_fitness = max(recent_fitnesses)
            recent_best = max(recent_fitnesses[-10:])
            
            if abs(best_fitness - recent_best) < self.early_stopping_tolerance:
                old_noise = self.current_noise_std
                self.current_noise_std = max(self.min_noise_std, 
                                           self.current_noise_std * self.noise_decrease_factor)
                if self.current_noise_std != old_noise:
                    print(f"   Stagnation detected - decreasing noise from {old_noise:.3f} to {self.current_noise_std:.3f}")
    
    def check_adaptive_switch(self):
        if len(self.fitness_history) < self.adaptive_threshold:
            return
        
        recent_fitnesses = [h['best_fitness'] for h in self.fitness_history[-self.adaptive_threshold:]]
        if len(recent_fitnesses) >= self.adaptive_threshold:
            best_fitness_in_window = max(recent_fitnesses)
            earliest_fitness_in_window = recent_fitnesses[0]
            
            improvement = best_fitness_in_window - earliest_fitness_in_window
            
            print(f"   Debug: Generation {len(self.fitness_history)}, improvement over last {self.adaptive_threshold} gens: {improvement:.6f}, tolerance: {self.early_stopping_tolerance}")
            
            if improvement < self.early_stopping_tolerance and not self.adaptive_mode:
                self.adaptive_mode = True
                print(f"   Switching to adaptive optimization mode at generation {len(self.fitness_history)} - no significant improvement in {self.adaptive_threshold} generations")
    
    def create_next_generation(self):
        new_population = []
        
        best_individual = max(self.population, key=lambda x: x['fitness'])
        new_population.append(copy.deepcopy(best_individual))
        
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            
            child1, child2 = self.crossover(parent1, parent2)
            
            self.mutate(child1)
            self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        self.population = new_population[:self.population_size]
        
        self.add_noise_to_population()
    
    def run_optimization(self, initial_naca: str = None):
        print("Starting AF512 Genetic Algorithm Optimization")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.generations}")
        print(f"Flight conditions:")
        print(f"   - Wingspan: {self.wingspan} ft")
        print(f"   - Chord: {self.chord} ft")
        print(f"   - Airspeed: {self.airspeed} mph")
        print(f"   - Reynolds number: {self.reynolds_number:.0f}")
        print(f"Alpha range: {self.alpha_range}")
        
        self.initialize_population(initial_naca)
        
        best_fitness_history = []
        generations_without_improvement = 0
        best_fitness_ever = 0.0
        
        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}/{self.generations}")
            
            self.evaluate_population()
            
            fitnesses = [ind['fitness'] for ind in self.population]
            best_fitness = float(max(fitnesses))
            avg_fitness = float(np.mean(fitnesses))
            best_individual = max(self.population, key=lambda x: x['fitness'])
            
            if best_fitness > best_fitness_ever + self.early_stopping_tolerance:
                best_fitness_ever = best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            best_fitness_history.append(best_fitness)
            
            print(f"Best L/D: {best_fitness:.3f}")
            print(f"Average L/D: {avg_fitness:.3f}")
            print(f"Generations without improvement: {generations_without_improvement}/{self.early_stopping_patience}")
            
            if generations_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping triggered! No improvement for {self.early_stopping_patience} generations.")
                print(f"Best L/D achieved: {best_fitness_ever:.3f}")
                break
            
            self.fitness_history.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_individual': copy.deepcopy(best_individual)
            })
            
            self.adjust_noise_dynamically()
            
            self.check_adaptive_switch()
            
            if self.image_callback:
                try:
                    self.image_callback(best_individual, generation + 1, best_fitness)
                except Exception as e:
                    print(f"Image capture failed for generation {generation + 1}: {e}")
            
            if generation < self.generations - 1:
                self.create_next_generation()
        
        self.evaluate_population()
        final_best = max(self.population, key=lambda x: x['fitness'])
        
        print(f"\nOptimization Complete!")
        if self.target_lift is not None:
            print(f"Target lift constraint: CL ≥ {self.target_lift}")
            if final_best['aerodynamic_data']['CL'] >= self.target_lift:
                print(f"Target lift achieved! CL = {final_best['aerodynamic_data']['CL']:.3f}")
            else:
                print(f"Target lift not achieved. CL = {final_best['aerodynamic_data']['CL']:.3f}")
        
        print(f"Best L/D ratio: {final_best['fitness']:.3f}")
        print(f"Peak L/D at: {final_best['aerodynamic_data']['peak_alpha']:.1f}°")
        print(f"CL: {final_best['aerodynamic_data']['CL']:.3f}")
        print(f"CD: {final_best['aerodynamic_data']['CD']:.4f}")
        print(f"Area under L/D curve: {final_best['aerodynamic_data']['area_under_curve']:.3f}")
        
        if self.adaptive_mode:
            print(f"Optimization mode: Adaptive (area under L/D curve preferred)")
        else:
            print(f"Optimization mode: Standard (peak L/D optimization)")
        
        if len(self.fitness_history) < self.generations:
            print(f"Early stopping: Completed {len(self.fitness_history)} generations out of {self.generations}")
        
        return final_best
    
    def plot_optimization_history(self):
        if not self.fitness_history:
            print("No optimization history to plot")
            return
        
        generations = [h['generation'] for h in self.fitness_history]
        best_fitnesses = [h['best_fitness'] for h in self.fitness_history]
        avg_fitnesses = [h['avg_fitness'] for h in self.fitness_history]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(generations, best_fitnesses, 'b-', linewidth=2, label='Best L/D')
        plt.plot(generations, avg_fitnesses, 'r--', linewidth=2, label='Average L/D')
        plt.xlabel('Generation')
        plt.ylabel('L/D Ratio')
        plt.title('Optimization History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if self.fitness_history:
            best_individual = self.fitness_history[-1]['best_individual']
            if best_individual['xy_coordinates']:
                x_coords, y_coords = best_individual['xy_coordinates']
                
                plt.subplot(2, 2, 2)
                plt.plot(x_coords, y_coords, 'b-', linewidth=2)
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.title('Best Optimized Airfoil')
                plt.grid(True, alpha=0.3)
                plt.axis('equal')
        
        plt.subplot(2, 2, 3)
        final_fitnesses = [ind['fitness'] for ind in self.population]
        plt.hist(final_fitnesses, bins=20, alpha=0.7, color='green')
        plt.xlabel('L/D Ratio')
        plt.ylabel('Frequency')
        plt.title('Final Population Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        if self.fitness_history:
            best_individual = self.fitness_history[-1]['best_individual']
            af512_data = best_individual['af512_data']
            x_points = np.linspace(0, 1, self.num_points)
            plt.plot(x_points, af512_data[:, 0], 'r-', linewidth=2, label='Upper Surface', alpha=0.8)
            plt.plot(x_points, af512_data[:, 1], 'b-', linewidth=2, label='Lower Surface', alpha=0.8)
            plt.xlabel('Chord Position (0-1)')
            plt.ylabel('Distance from Chord Line')
            plt.title('Best AF512 Format')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('af512_optimization_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_early_stopping_stats(self):
        if not self.fitness_history:
            return None
        
        best_fitnesses = [h['best_fitness'] for h in self.fitness_history]
        max_fitness = max(best_fitnesses)
        max_fitness_gen = best_fitnesses.index(max_fitness) + 1
        
        return {
            'total_generations': len(self.fitness_history),
            'max_generations': self.generations,
            'early_stopped': len(self.fitness_history) < self.generations,
            'best_fitness': max_fitness,
            'best_fitness_generation': max_fitness_gen,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_tolerance': self.early_stopping_tolerance
        }
    
    def save_results(self, filename: str = 'af512_optimization_results.json'):
        if not self.fitness_history:
            print("No results to save")
            return
        
        results = {
            'parameters': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'wingspan': self.wingspan,
                'chord': self.chord,
                'airspeed': self.airspeed,
                'reynolds_number': self.reynolds_number,
                'alpha_range': self.alpha_range
            },
            'optimization_history': [
                {
                    'generation': h['generation'],
                    'best_fitness': float(h['best_fitness']),
                    'avg_fitness': float(h['avg_fitness'])
                }
                for h in self.fitness_history
            ],
            'final_population': [
                {
                    'fitness': float(ind['fitness']),
                    'aerodynamic_data': {
                        'CL': float(ind['aerodynamic_data']['CL']),
                        'CD': float(ind['aerodynamic_data']['CD']),
                        'L/D': float(ind['aerodynamic_data']['L/D']),
                        'alpha': float(ind['aerodynamic_data']['alpha'])
                    }
                }
                for ind in self.population
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")

GeneticNACAOptimizer = GeneticAF512Optimizer

def main():
    print("AF512 Airfoil Genetic Algorithm Optimizer")
    print("=" * 50)
    print("Fixed Flight Conditions:")
    print("   - Wingspan: 6.0 ft")
    print("   - Chord: 0.8 ft") 
    print("   - Airspeed: 30.0 mph")
    print("   - Reynolds number: ~223,000")
    print("=" * 50)
    
    initial_naca = input("Enter initial NACA code (e.g., 4012) or press Enter for random: ").strip()
    if not initial_naca:
        initial_naca = None
    
    try:
        population_size = int(input("Population size (default 50): ") or "50")
        generations = int(input("Number of generations (default 50): ") or "50")
    except ValueError:
        print("Using default values")
        population_size = 50
        generations = 50
    
    optimizer = GeneticAF512Optimizer(
        population_size=population_size,
        generations=generations,
        wingspan=6.0,
        chord=0.8,
        airspeed=30.0
    )
    
    best_airfoil = optimizer.run_optimization(initial_naca)
    
    optimizer.plot_optimization_history()
    
    optimizer.save_results()
    
    if initial_naca:
        print(f"\nComparison:")
        initial_individual = optimizer.create_individual_from_naca(initial_naca)
        initial_fitness = optimizer.evaluate_fitness(initial_individual)
        print(f"Initial NACA {initial_naca}: L/D = {initial_fitness:.3f}")
        print(f"Optimized AF512: L/D = {best_airfoil['fitness']:.3f}")
        improvement = ((best_airfoil['fitness'] - initial_fitness) / initial_fitness) * 100
        print(f"Improvement: {improvement:.1f}%")

if __name__ == "__main__":
    main()

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
                 generations: int = 1000,
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
        
        # Two-stage optimization parameters
        self.stage1_complete = False
        self.stage1_best_peak_ld = 0.0
        self.stage2_generations = 1000  # Additional generations for stage 2
        self.peak_ld_preservation_threshold = 0.98  # Must maintain 98% of original peak L/D
        
        # Store the non-area optimized airfoil for comparison
        self.stage1_final_airfoil = None
        
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
    
    def create_individual_from_airfoil(self, airfoil_code: str) -> dict:
        """Create individual from airfoil code (supports both NACA and Selig formats)"""
        try:
            from naca_trainer import airfoil_to_af512
            af512_data = airfoil_to_af512(airfoil_code, self.num_points)
            return {
                'af512_data': af512_data,
                'fitness': None,
                'xy_coordinates': None,
                'aerodynamic_data': None
            }
        except Exception as e:
            print(f"Error creating from airfoil {airfoil_code}: {e}")
            return self.create_individual()
    
    def create_individual_from_naca(self, naca_code: str) -> dict:
        """Legacy method - now calls create_individual_from_airfoil"""
        return self.create_individual_from_airfoil(naca_code)
    
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
        if not self.stage1_complete:
            # Stage 1: Optimize for peak L/D only (no falloff penalty)
            return aero_data['L/D']
        else:
            # Stage 2: Optimize for area under L/D curve + falloff while preserving peak L/D
            current_peak_ld = aero_data['L/D']
            
            # Check if peak L/D is preserved within threshold
            if current_peak_ld >= self.stage1_best_peak_ld * self.peak_ld_preservation_threshold:
                # Stage 2: Reward based on area improvement and falloff ratio
                # Calculate the improvement in area from Stage 1 to current
                current_area = aero_data['area_under_curve']
                stage1_area = self.stage1_final_airfoil['aerodynamic_data']['area_under_curve'] if self.stage1_final_airfoil else 0
                
                # Area improvement: difference between current and Stage 1 area
                area_improvement = current_area - stage1_area
                
                # Calculate falloff ratio (L/D at 10° / peak L/D)
                falloff_ratio = 1.0  # Default to 1.0 (no falloff)
                ld_values = aero_data.get('ld_values', [])
                if ld_values and len(ld_values) >= 21:
                    peak_ld = max(ld_values)
                    ld_at_10deg = ld_values[15]  # Index 15 = 10 degrees
                    if peak_ld > 0:
                        falloff_ratio = ld_at_10deg / peak_ld
                
                # Reward = area improvement × 10 × falloff ratio
                # This rewards both area improvement and maintaining good falloff characteristics
                reward = area_improvement * 10.0 * falloff_ratio
                
                # Add small bonus for peak L/D preservation (minimal weight)
                peak_bonus = current_peak_ld * 0.01
                
                return reward + peak_bonus
            else:
                # Strong penalty if peak L/D drops below threshold - we must conserve the peak
                penalty_factor = 0.01  # Very strong penalty
                return current_peak_ld * penalty_factor
    
    def _calculate_falloff_penalty(self, aero_data: dict) -> float:
        """Calculate penalty for aggressive L/D falloff beyond peak"""
        try:
            # Get L/D values at different angles of attack
            ld_values = aero_data.get('ld_values', [])
            if not ld_values or len(ld_values) < 21:  # Need full alpha range
                return 0.0
            
            # Find peak L/D and its index
            peak_ld = max(ld_values)
            peak_idx = ld_values.index(peak_ld)
            
            # Check L/D at 10 degrees (index 15 in -5 to 15 range)
            target_idx = 15
            if target_idx < len(ld_values):
                ld_at_10deg = ld_values[target_idx]
                
                # Calculate how much L/D has fallen from peak
                if peak_ld > 0:
                    falloff_ratio = ld_at_10deg / peak_ld
                    
                    # If L/D at 10° is less than 90% of peak, apply penalty
                    if falloff_ratio < 0.9:  # 10% falloff threshold
                        # Calculate penalty based on how much it falls below 90%
                        penalty_strength = 50.0  # Strong penalty for aggressive falloff
                        penalty = penalty_strength * (0.9 - falloff_ratio)
                        
                        # Log the falloff penalty for debugging
                        if penalty > 5.0:  # Only log significant penalties
                            stage_info = "Stage 2" if self.stage1_complete else "Stage 1"
                            print(f"   [{stage_info}] L/D falloff penalty: {penalty:.1f} (peak: {peak_ld:.2f}, at 10°: {ld_at_10deg:.2f}, ratio: {falloff_ratio:.3f})")
                        
                        return penalty
                    
                    # Bonus for maintaining L/D above 90% at 10°
                    if falloff_ratio >= 0.9:
                        bonus = 10.0 * (falloff_ratio - 0.9)  # Small bonus for better performance
                        if bonus > 0.5:  # Only log significant bonuses
                            stage_info = "Stage 2" if self.stage1_complete else "Stage 1"
                            print(f"   [{stage_info}] L/D falloff bonus: {bonus:.1f} (ratio: {falloff_ratio:.3f})")
                        return -bonus  # Negative penalty = bonus
            
            return 0.0
            
        except Exception as e:
            print(f"Error calculating falloff penalty: {e}")
            return 0.0
    
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
    
    def create_next_generation_stage2(self):
        new_population = []
        
        best_individual = max(self.population, key=lambda x: x['fitness'])
        new_population.append(copy.deepcopy(best_individual))
        
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutate children for stage 2 (no noise)
            self.mutate(child1)
            self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        self.population = new_population[:self.population_size]
        # Add 0.5% noise in Stage 2 for exploration
        self.add_noise_to_population(noise_std=0.005)
    
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
        
        # Stage 1: Optimize for peak L/D
        print(f"\n" + "="*60)
        print(f"STAGE 1: Optimizing for Peak L/D")
        print(f"="*60)
        print(f"Focus: Maximize peak L/D ratio only")
        print(f"No falloff penalties applied in this stage")
        
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
            
            if self.image_callback:
                try:
                    self.image_callback(best_individual, generation + 1, best_fitness)
                except Exception as e:
                    print(f"Image capture failed for generation {generation + 1}: {e}")
            
            if generation < self.generations - 1:
                self.create_next_generation()
        
        # Stage 1 complete - record best peak L/D
        self.stage1_complete = True
        self.stage1_best_peak_ld = best_fitness_ever
        
        # Store the Stage 1 final airfoil (non-area optimized) for comparison
        self.stage1_final_airfoil = copy.deepcopy(best_individual)
        
        print(f"\n" + "="*60)
        print(f"STAGE 1 COMPLETE: Peak L/D = {self.stage1_best_peak_ld:.3f}")
        print(f"Switching to Stage 2: Optimizing for area under L/D curve + falloff characteristics")
        print(f"Must maintain peak L/D ≥ {self.stage1_best_peak_ld * self.peak_ld_preservation_threshold:.3f} ({self.peak_ld_preservation_threshold*100:.0f}% of best)")
        print(f"Falloff penalties now active: L/D must stay ≥90% of peak at 10° angle of attack")
        print(f"="*60)
        
        # Stage 2: Optimize for area under L/D curve + falloff while preserving peak L/D
        print(f"\nStage 2: Optimizing for area under L/D curve + falloff characteristics (preserving peak L/D)")
        print(f"Additional generations: {self.stage2_generations}")
        print(f"Falloff optimization: L/D must maintain ≥90% of peak value at 10° angle of attack")
        print(f"Stage 2 early stopping patience: 50 generations")
        
        # Reset early stopping for stage 2 with 50 generation patience
        generations_without_improvement = 0
        stage2_best_fitness = best_fitness_ever
        stage2_early_stopping_patience = 50  # Fixed 50 generation patience for stage 2
        
        for generation in range(self.stage2_generations):
            print(f"\nStage 2 Generation {generation + 1}/{self.stage2_generations}")
            
            self.evaluate_population()
            
            fitnesses = [ind['fitness'] for ind in self.population]
            best_fitness = float(max(fitnesses))
            avg_fitness = float(np.mean(fitnesses))
            best_individual = max(self.population, key=lambda x: x['fitness'])
            
            # Check if we're still improving in stage 2
            if best_fitness > stage2_best_fitness + self.early_stopping_tolerance:
                stage2_best_fitness = best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            print(f"Best fitness: {best_fitness:.3f}")
            print(f"Average fitness: {avg_fitness:.3f}")
            print(f"Generations without improvement: {generations_without_improvement}/{stage2_early_stopping_patience}")
            
            # Early stopping for stage 2 with 50 generation patience
            if generations_without_improvement >= stage2_early_stopping_patience:
                print(f"Stage 2 early stopping triggered! No improvement for {stage2_early_stopping_patience} generations.")
                break
            
            # Add to fitness history with stage indicator
            self.fitness_history.append({
                'generation': len(self.fitness_history) + 1,
                'stage': 2,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_individual': copy.deepcopy(best_individual)
            })
            
            if self.image_callback:
                try:
                    self.image_callback(best_individual, len(self.fitness_history), best_fitness)
                except Exception as e:
                    print(f"Image capture failed for stage 2 generation {generation + 1}: {e}")
            
            if generation < self.stage2_generations - 1:
                # Create next generation WITHOUT adding noise (clean optimization)
                self.create_next_generation_stage2()
        
        # Final evaluation
        self.evaluate_population()
        final_best = max(self.population, key=lambda x: x['fitness'])
        
        print(f"\nOptimization Complete!")
        print(f"Stage 1: Peak L/D = {self.stage1_best_peak_ld:.3f}")
        print(f"Stage 2: Final fitness = {final_best['fitness']:.3f}")
        
        if self.target_lift is not None:
            print(f"Target lift constraint: CL ≥ {self.target_lift}")
            if final_best['aerodynamic_data']['CL'] >= self.target_lift:
                print(f"Target lift achieved! CL = {final_best['aerodynamic_data']['CL']:.3f}")
            else:
                print(f"Target lift not achieved. CL = {final_best['aerodynamic_data']['CL']:.3f}")
        
        print(f"Best L/D ratio: {final_best['aerodynamic_data']['L/D']:.3f}")
        print(f"Peak L/D at: {final_best['aerodynamic_data']['peak_alpha']:.1f}°")
        print(f"CL: {final_best['aerodynamic_data']['CL']:.3f}")
        print(f"CD: {final_best['aerodynamic_data']['CD']:.4f}")
        print(f"Area under L/D curve: {final_best['aerodynamic_data']['area_under_curve']:.3f}")
        
        # Check if peak L/D was preserved
        final_peak_ld = final_best['aerodynamic_data']['L/D']
        if final_peak_ld >= self.stage1_best_peak_ld * self.peak_ld_preservation_threshold:
            print(f"✓ Peak L/D preserved: {final_peak_ld:.3f} ≥ {self.stage1_best_peak_ld * self.peak_ld_preservation_threshold:.3f}")
        else:
            print(f"⚠ Peak L/D not preserved: {final_peak_ld:.3f} < {self.stage1_best_peak_ld * self.peak_ld_preservation_threshold:.3f}")
        
        print(f"Optimization mode: Two-stage (peak L/D → area under curve)")
        
        if len(self.fitness_history) < (self.generations + self.stage2_generations):
            print(f"Early stopping: Completed {len(self.fitness_history)} generations out of {self.generations + self.stage2_generations}")
        
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
    
    def get_stage1_final_airfoil(self):
        """Get the final airfoil from Stage 1 (non-area optimized) for comparison"""
        return self.stage1_final_airfoil
    
    def get_early_stopping_stats(self):
        if not self.fitness_history:
            return None
        
        best_fitnesses = [h['best_fitness'] for h in self.fitness_history]
        max_fitness = max(best_fitnesses)
        max_fitness_gen = best_fitnesses.index(max_fitness) + 1
        
        # Separate stage 1 and stage 2 statistics
        stage1_generations = [h for h in self.fitness_history if 'stage' not in h or h.get('stage') == 1]
        stage2_generations = [h for h in self.fitness_history if h.get('stage') == 2]
        
        stage1_best = max([h['best_fitness'] for h in stage1_generations]) if stage1_generations else 0
        stage2_best = max([h['best_fitness'] for h in stage2_generations]) if stage2_generations else 0
        
        return {
            'total_generations': len(self.fitness_history),
            'max_generations': self.generations + self.stage2_generations,
            'early_stopped': len(self.fitness_history) < (self.generations + self.stage2_generations),
            'best_fitness': max_fitness,
            'best_fitness_generation': max_fitness_gen,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_tolerance': self.early_stopping_tolerance,
            'stage1_complete': self.stage1_complete,
            'stage1_best_peak_ld': self.stage1_best_peak_ld,
            'stage1_generations': len(stage1_generations),
            'stage2_generations': len(stage2_generations),
            'stage1_best_fitness': stage1_best,
            'stage2_best_fitness': stage2_best,
            'peak_ld_preserved': (self.stage1_best_peak_ld > 0 and 
                                stage2_best >= self.stage1_best_peak_ld * self.peak_ld_preservation_threshold),
            'stage1_final_airfoil': self.get_stage1_final_airfoil()
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

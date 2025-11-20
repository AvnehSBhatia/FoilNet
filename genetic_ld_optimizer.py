import torch
import numpy as np
import matplotlib.pyplot as plt
from naca_trainer import AF512toXYNet, predict_xy, naca_to_af, naca4_to_xy
import random
import copy
from typing import List, Tuple, Optional
import json
import os

CUSTOM_CLIP_RANGES: Optional[List[dict]] = [
    {'x_range': (0.0, 1.0), 'top_clip': (0.001, 0.36), 'bottom_clip': (0.001, 0.36)},
]
# User-adjustable alpha (angle of attack) settings
ALPHA_RANGE: Tuple[float, float] = (-10, 20)  # (min_alpha, max_alpha) in degrees
ALPHA_INCREMENT: float = 1  # Increment in degrees (e.g., 0.1, 0.5, 1.0)

AIRFOIL_MODES: Tuple[str, ...] = ("normal", "symmetric", "flat")
DEFAULT_AIRFOIL_MODE: str = "normal"


def apply_surface_clipping(
    upper_dist: np.ndarray,
    lower_dist: np.ndarray,
    x_points: np.ndarray
) -> None:
    # Apply default clipping first
    default_top_min, default_top_max = 0.001, 0.16
    default_bottom_min, default_bottom_max = -0.16, -0.001
    np.clip(upper_dist, default_top_min, default_top_max, out=upper_dist)
    np.clip(lower_dist, default_bottom_min, default_bottom_max, out=lower_dist)

    # Apply custom clipping ranges on top (overrides defaults in those regions)
    if CUSTOM_CLIP_RANGES:
        for clip_config in CUSTOM_CLIP_RANGES:
            x_min, x_max = clip_config['x_range']
            top_clip_min, top_clip_max = clip_config['top_clip']
            bottom_clip_min, bottom_clip_max = clip_config['bottom_clip']

            mask = (x_points >= x_min) & (x_points <= x_max)
            if np.any(mask):
                upper_dist[mask] = np.clip(upper_dist[mask], top_clip_min, top_clip_max)
                lower_dist[mask] = np.clip(lower_dist[mask], -bottom_clip_max, -bottom_clip_min)
        return

    # Legacy single-range support for backwards compatibility
    legacy_x_range = globals().get('CUSTOM_CLIP_X_RANGE')
    legacy_limits = globals().get('CUSTOM_CLIP_LIMITS')
    if legacy_x_range and legacy_limits:
        x_min, x_max = legacy_x_range
        clip_min, clip_max = legacy_limits
        mask = (x_points >= x_min) & (x_points <= x_max)
        if np.any(mask):
            upper_dist[mask] = np.clip(upper_dist[mask], clip_min, clip_max)
            lower_dist[mask] = np.clip(lower_dist[mask], -clip_max, -clip_min)


def apply_clipping_to_af512(
    af512_data: np.ndarray,
    x_points: np.ndarray
) -> None:
    apply_surface_clipping(af512_data[:, 0], af512_data[:, 1], x_points)

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
                 early_stopping_tolerance: float = 0.001,
                 airfoil_mode: str = DEFAULT_AIRFOIL_MODE):
        
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
        if airfoil_mode not in AIRFOIL_MODES:
            raise ValueError(f"Invalid airfoil_mode '{airfoil_mode}'. Choose from {AIRFOIL_MODES}.")
        self.airfoil_mode = airfoil_mode
        
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
        
        # Create alpha_range array from user settings
        alpha_min, alpha_max = ALPHA_RANGE
        num_steps = int((alpha_max - alpha_min) / ALPHA_INCREMENT) + 1
        self.alpha_range = np.linspace(alpha_min, alpha_max, num_steps)
        
        # Four-stage optimization parameters
        self.stage1_complete = False
        self.stage2_complete = False
        self.stage3_complete = False
        self.stage1_best_peak_ld = 0.0
        self.stage2_best_peak_cl = 0.0
        self.stage3_best_avg_cl = 0.0
        self.stage2_generations = 1000  # Additional generations for stage 2
        self.stage3_generations = 1000  # Additional generations for stage 3
        self.stage4_generations = 1000  # Additional generations for stage 4
        self.peak_ld_preservation_threshold = 0.98  # Must maintain 98% of original peak L/D (2% max drop)
        self.peak_cl_preservation_threshold = 0.98  # Must maintain 98% of original peak CL (2% max drop)
        self.avg_cl_preservation_threshold = 0.98  # Must maintain 98% of stage 3 avg CL (2% max drop)
        
        # Store the stage final airfoils for comparison
        self.stage1_final_airfoil = None
        self.stage2_final_airfoil = None
        self.stage3_final_airfoil = None
        
    def make_individual(self) -> dict:
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
        
        apply_surface_clipping(upper_dist, lower_dist, x_points)
        
        upper_dist[0] = 0.0
        lower_dist[0] = 0.0
        
        if self.airfoil_mode == "normal":
            leading_edge_thickness = upper_dist[-1] - lower_dist[-1]
            if leading_edge_thickness < 0.01:
                center = (upper_dist[-1] + lower_dist[-1]) / 2
                upper_dist[-1] = center + 0.005
                lower_dist[-1] = center - 0.005
        
        af512_data = np.column_stack([upper_dist, lower_dist])
        
        individual = {
            'af512_data': af512_data,
            'fitness': None,
            'xy_coordinates': None,
            'aerodynamic_data': None
        }
        self._apply_mode_to_ind(individual)
        return individual
    
    def make_from_code(self, airfoil_code: str) -> dict:
        """Create individual from airfoil code (supports both NACA and Selig formats)"""
        try:
            from naca_trainer import airfoil_to_af512
            af512_data = airfoil_to_af512(airfoil_code, self.num_points)
            individual = {
                'af512_data': af512_data,
                'fitness': None,
                'xy_coordinates': None,
                'aerodynamic_data': None
            }
            self._apply_mode_to_ind(individual)
            return individual
        except Exception as e:
            print(f"Error creating from airfoil {airfoil_code}: {e}")
            return self.make_individual()
    
    def make_from_naca(self, naca_code: str) -> dict:
        """Legacy method - now calls make_from_code"""
        return self.make_from_code(naca_code)
    
    def init_population(self, initial_naca: str = None):
        print(f"Initializing population of {self.population_size} individuals...")
        self.population = [self.make_individual() for _ in range(self.population_size)]
        
        if initial_naca:
            print(f"Using initial NACA: {initial_naca}")
            self.population[0] = self.make_from_naca(initial_naca)
    
    def af512_to_xy(self, af512_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            pred_x, pred_y = predict_xy(self.model, af512_data, self.device)
            
            pred_x = np.clip(pred_x, 0, 1)
            
            return pred_x, pred_y
            
        except Exception as e:
            print(f"Error converting AF512 to coordinates: {e}")
            x = np.linspace(0, 1, self.num_points)
            y = np.zeros_like(x)
            return x, y
    
    def eval_aero(self, x_coords: np.ndarray, y_coords: np.ndarray) -> dict:
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
            cl_values = []
            
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
                    cl_values.append(cl)
                    
                    if ld_ratio > best_ld:
                        best_ld = ld_ratio
                        best_alpha = alpha
                        best_cl = cl
                        best_cd = cd
                
                area_under_curve = np.trapz(ld_values, self.alpha_range)
                
                # Calculate average L/D over 0-12.5 degrees
                # Find indices closest to 0 and 12.5 degrees dynamically
                start_idx = np.argmin(np.abs(self.alpha_range - 0.0))
                end_alpha = min(12.5, self.alpha_range[-1])  # Don't exceed max alpha
                end_idx = np.argmin(np.abs(self.alpha_range - end_alpha))
                if end_idx < len(ld_values) and start_idx <= end_idx:
                    ld_0_to_12_5 = ld_values[start_idx:end_idx+1]
                    average_ld_0_to_12_5 = np.mean(ld_0_to_12_5)
                else:
                    average_ld_0_to_12_5 = 0.0
                
                return {
                    'CL': best_cl,
                    'CD': best_cd,
                    'L/D': best_ld,
                    'alpha': best_alpha,
                    'peak_alpha': best_alpha,
                    'area_under_curve': area_under_curve,
                    'ld_values': ld_values,
                    'cl_values': cl_values,
                    'average_ld_0_to_12_5': average_ld_0_to_12_5
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
                    'ld_values': [0.0] * len(self.alpha_range),
                    'cl_values': [0.0] * len(self.alpha_range),
                    'average_ld_0_to_12_5': 0.0
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
                'ld_values': [0.0] * len(self.alpha_range),
                'cl_values': [0.0] * len(self.alpha_range),
                'average_ld_0_to_12_5': 0.0
            }
    
    def eval_fitness(self, individual: dict) -> float:
        try:
            self._apply_mode_to_ind(individual)
            x_coords, y_coords = self.af512_to_xy(individual['af512_data'])
            individual['xy_coordinates'] = (x_coords, y_coords)
            
            aero_data = self.eval_aero(x_coords, y_coords)
            individual['aerodynamic_data'] = aero_data
            
            if self.target_lift is not None:
                if aero_data['CL'] >= self.target_lift:
                    fitness = self._stage_fitness(aero_data)
                else:
                    lift_penalty = (self.target_lift - aero_data['CL']) * 1000
                    fitness = -lift_penalty
            else:
                fitness = self._stage_fitness(aero_data)
            
            individual['fitness'] = fitness
            return fitness
            
        except Exception as e:
            print(f"Error evaluating fitness: {e}")
            individual['fitness'] = 0.0
            return 0.0
    
    def _stage_fitness(self, aero_data: dict) -> float:
        if not self.stage1_complete:
            # Stage 1: Optimize for peak L/D only
            return aero_data['L/D']
        elif not self.stage2_complete:
            # Stage 2: Optimize for peak CL at peak L/D while preserving peak L/D within 2%
            current_peak_ld = aero_data['L/D']
            
            # Check if peak L/D is preserved within 2% threshold
            if current_peak_ld >= self.stage1_best_peak_ld * self.peak_ld_preservation_threshold:
                # Return peak CL at the peak L/D angle multiplied by 100
                return aero_data['CL'] * 100
            else:
                # Strong penalty if peak L/D drops below 2% threshold
                penalty_factor = 0.01  # Very strong penalty
                return current_peak_ld * penalty_factor
        elif not self.stage3_complete:
            # Stage 3: Optimize for average CL over ±4° range around peak L/D
            current_peak_ld = aero_data['L/D']
            current_peak_cl = aero_data['CL']
            
            # Check if both peak L/D and peak CL are preserved within 2% threshold
            ld_preserved = current_peak_ld >= self.stage1_best_peak_ld * self.peak_ld_preservation_threshold
            cl_preserved = current_peak_cl >= self.stage2_best_peak_cl * self.peak_cl_preservation_threshold
            
            if ld_preserved and cl_preserved:
                # Calculate average CL over ±4° range around peak L/D
                cl_values = aero_data.get('cl_values', [])
                
                if not cl_values or len(cl_values) != len(self.alpha_range):
                    return 0.0
                
                # Find the peak L/D angle index
                ld_values = aero_data.get('ld_values', [])
                peak_ld_idx = ld_values.index(max(ld_values))
                
                # Calculate ±4° range around peak L/D dynamically
                # Convert 4 degrees to number of indices based on increment
                indices_per_4deg = int(np.ceil(4.0 / ALPHA_INCREMENT))
                start_idx = max(0, peak_ld_idx - indices_per_4deg)
                end_idx = min(len(cl_values) - 1, peak_ld_idx + indices_per_4deg)
                
                # Calculate average CL over the ±4° range
                cl_range = cl_values[start_idx:end_idx+1]
                avg_cl = np.mean(cl_range)
                
                return avg_cl
            else:
                # Strong penalty if either peak L/D or peak CL drops below 2% threshold
                penalty_factor = 0.01  # Very strong penalty
                return min(current_peak_ld, current_peak_cl) * penalty_factor
        else:
            # Stage 4: Optimize for average L/D over ±4° range around peak L/D
            current_peak_ld = aero_data['L/D']
            current_peak_cl = aero_data['CL']
            
            # Check if all previous stage values are preserved within 2% threshold
            ld_preserved = current_peak_ld >= self.stage1_best_peak_ld * self.peak_ld_preservation_threshold
            cl_preserved = current_peak_cl >= self.stage2_best_peak_cl * self.peak_cl_preservation_threshold
            
            if ld_preserved and cl_preserved:
                # Calculate average L/D over ±4° range around peak L/D
                ld_values = aero_data.get('ld_values', [])
                
                if not ld_values or len(ld_values) != len(self.alpha_range):
                    return 0.0
                
                # Find the peak L/D angle index
                peak_ld_idx = ld_values.index(max(ld_values))
                
                # Calculate ±4° range around peak L/D dynamically
                # Convert 4 degrees to number of indices based on increment
                indices_per_4deg = int(np.ceil(4.0 / ALPHA_INCREMENT))
                start_idx = max(0, peak_ld_idx - indices_per_4deg)
                end_idx = min(len(ld_values) - 1, peak_ld_idx + indices_per_4deg)
                
                # Calculate average L/D over the ±4° range
                ld_range = ld_values[start_idx:end_idx+1]
                avg_ld = np.mean(ld_range)
                
                return avg_ld
            else:
                # Strong penalty if either peak L/D or peak CL drops below 2% threshold
                penalty_factor = 0.01  # Very strong penalty
                return min(current_peak_ld, current_peak_cl) * penalty_factor
    
    def _falloff_penalty(self, aero_data: dict) -> float:
        """Calculate penalty for aggressive L/D falloff beyond peak"""
        try:
            # Get L/D values at different angles of attack
            ld_values = aero_data.get('ld_values', [])
            if not ld_values or len(ld_values) != len(self.alpha_range):
                return 0.0
            
            # Find peak L/D and its index
            peak_ld = max(ld_values)
            peak_idx = ld_values.index(peak_ld)
            
            # Check L/D at 10 degrees (find closest index dynamically)
            target_alpha = 10.0
            if target_alpha <= self.alpha_range[-1]:
                target_idx = np.argmin(np.abs(self.alpha_range - target_alpha))
                if target_idx < len(ld_values):
                    ld_at_10deg = ld_values[target_idx]
                else:
                    return 0.0
            else:
                return 0.0
            
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

    def _apply_mode(self, af512_data: np.ndarray) -> None:
        """Apply mode-specific constraints to AF512 representation."""
        if self.airfoil_mode == "symmetric":
            upper = np.clip(af512_data[:, 0], 0.0, None)
            af512_data[:, 0] = upper
            af512_data[:, 1] = -upper
        elif self.airfoil_mode == "flat":
            af512_data[:, 0] = np.clip(af512_data[:, 0], 0.0, None)
            af512_data[:, 1] = 0.0
        else:
            af512_data[:, 0] = np.clip(af512_data[:, 0], 0.0, None)
            af512_data[:, 1] = np.clip(af512_data[:, 1], None, 0.0)
    
    def _apply_mode_to_ind(self, individual: dict) -> None:
        af512_data = individual.get('af512_data')
        if af512_data is None:
            return
        self._apply_mode(af512_data)
        af512_data[0, :] = 0.0
    
    def eval_population(self):
        print("Evaluating population fitness...")
        for i, individual in enumerate(self.population):
            self._apply_mode_to_ind(individual)
            if i % 10 == 0:
                print(f"   Evaluating individual {i+1}/{len(self.population)}")
            self.eval_fitness(individual)
    
    def pick_parents(self) -> Tuple[dict, dict]:
        tournament_size = 3
        
        def tournament_select():
            tournament = random.sample(self.population, tournament_size)
            return max(tournament, key=lambda x: x['fitness'])
        
        parent1 = tournament_select()
        parent2 = tournament_select()
        
        return parent1, parent2
    
    def blend(self, parent1: dict, parent2: dict) -> Tuple[dict, dict]:
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
        
        x_points = np.linspace(0, 1, self.num_points)
        for af512_data in [af512_1, af512_2]:
            apply_clipping_to_af512(af512_data, x_points)
            self._apply_mode(af512_data)
            af512_data[0, :] = 0.0

        child1['af512_data'] = af512_1
        child2['af512_data'] = af512_2
        self._apply_mode_to_ind(child1)
        self._apply_mode_to_ind(child2)
        
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
        
        x_points = np.linspace(0, 1, self.num_points)
        apply_clipping_to_af512(af512_data, x_points)
        self._apply_mode_to_ind(individual)
        
        from scipy.ndimage import gaussian_filter1d
        af512_data[:, 0] = gaussian_filter1d(af512_data[:, 0], sigma=3)
        af512_data[:, 1] = gaussian_filter1d(af512_data[:, 1], sigma=3)

        apply_clipping_to_af512(af512_data, x_points)
        self._apply_mode_to_ind(individual)
        
        if self.airfoil_mode == "normal":
            leading_edge_thickness = af512_data[-1, 0] - af512_data[-1, 1]
            if leading_edge_thickness < 0.01:
                center = (af512_data[-1, 0] + af512_data[-1, 1]) / 2
                af512_data[-1, 0] = center + 0.005
                af512_data[-1, 1] = center - 0.005
        
        self._apply_mode_to_ind(individual)
        
        individual['fitness'] = None
    
    def add_noise(self, individual, noise_std=0.01):
        if 'af512_data' in individual:
            noise = np.random.normal(0, noise_std, individual['af512_data'].shape)
            individual['af512_data'] += noise
            
            from scipy.ndimage import gaussian_filter1d
            individual['af512_data'][:, 0] = gaussian_filter1d(individual['af512_data'][:, 0], sigma=3)
            individual['af512_data'][:, 1] = gaussian_filter1d(individual['af512_data'][:, 1], sigma=3)
            
            x_points = np.linspace(0, 1, self.num_points)
            apply_clipping_to_af512(individual['af512_data'], x_points)
            self._apply_mode_to_ind(individual)
            
            individual['fitness'] = None
            individual['xy_coordinates'] = None
            individual['aerodynamic_data'] = None
    
    def shake_population(self, noise_std=None):
        if noise_std is None:
            noise_std = self.current_noise_std
        print(f"   Adding noise (std={noise_std:.3f}) to population...")
        for individual in self.population:
            self.add_noise(individual, noise_std)
    
    def tune_noise(self):
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
    
    def check_adaptive(self):
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
    
    def next_generation(self):
        new_population = []
        
        best_individual = max(self.population, key=lambda x: x['fitness'])
        elite = copy.deepcopy(best_individual)
        self._apply_mode_to_ind(elite)
        new_population.append(elite)
        
        while len(new_population) < self.population_size:
            parent1, parent2 = self.pick_parents()
            
            child1, child2 = self.blend(parent1, parent2)
            
            self.mutate(child1)
            self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        self.population = new_population[:self.population_size]
        
        self.shake_population()
    
    def next_generation_stage2(self):
        new_population = []
        
        best_individual = max(self.population, key=lambda x: x['fitness'])
        elite = copy.deepcopy(best_individual)
        self._apply_mode_to_ind(elite)
        new_population.append(elite)
        
        while len(new_population) < self.population_size:
            parent1, parent2 = self.pick_parents()
            
            child1, child2 = self.blend(parent1, parent2)
            
            # Mutate children for stage 2 (no noise)
            self.mutate(child1)
            self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        self.population = new_population[:self.population_size]
        # Add 0.5% noise in Stage 2 for exploration
        self.shake_population(noise_std=0.005)
    
    def run(self, initial_naca: str = None):
        print("Starting AF512 Genetic Algorithm Optimization")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.generations}")
        print(f"Flight conditions:")
        print(f"   - Wingspan: {self.wingspan} ft")
        print(f"   - Chord: {self.chord} ft")
        print(f"   - Airspeed: {self.airspeed} mph")
        print(f"   - Reynolds number: {self.reynolds_number:.0f}")
        print(f"Alpha range: {self.alpha_range}")
        
        self.init_population(initial_naca)
        
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
            
            self.eval_population()
            
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
            
            self.tune_noise()
            
            if self.image_callback:
                try:
                    self.image_callback(best_individual, generation + 1, best_fitness)
                except Exception as e:
                    print(f"Image capture failed for generation {generation + 1}: {e}")
            
            if generation < self.generations - 1:
                self.next_generation()
        
        # Stage 1 complete - record best peak L/D
        self.stage1_complete = True
        self.stage1_best_peak_ld = best_fitness_ever
        
        # Store the Stage 1 final airfoil for comparison
        self.stage1_final_airfoil = copy.deepcopy(best_individual)
        self._apply_mode_to_ind(self.stage1_final_airfoil)
        
        print(f"\n" + "="*60)
        print(f"STAGE 1 COMPLETE: Peak L/D = {self.stage1_best_peak_ld:.3f}")
        print(f"Switching to Stage 2: Optimizing for peak CL at peak L/D")
        print(f"Must maintain peak L/D ≥ {self.stage1_best_peak_ld * self.peak_ld_preservation_threshold:.3f} ({self.peak_ld_preservation_threshold*100:.0f}% of best)")
        print(f"Stage 2 focus: Maximize peak CL while preserving peak L/D within 2%")
        print(f"="*60)
        
        # Stage 2: Optimize for peak CL at peak L/D while preserving peak L/D
        print(f"\nStage 2: Optimizing for peak CL at peak L/D (preserving peak L/D)")
        print(f"Additional generations: {self.stage2_generations}")
        print(f"Peak CL optimization: Maximize CL at the peak L/D angle")
        print(f"Stage 2 early stopping patience: 50 generations")
        
        # Reset early stopping for stage 2 with 50 generation patience
        generations_without_improvement = 0
        stage2_best_fitness = best_fitness_ever
        stage2_early_stopping_patience = 50  # Fixed 50 generation patience for stage 2
        
        for generation in range(self.stage2_generations):
            print(f"\nStage 2 Generation {generation + 1}/{self.stage2_generations}")
            
            self.eval_population()
            
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
                self.next_generation_stage2()
        
        # Stage 2 complete - record best peak CL
        self.stage2_complete = True
        self.stage2_best_peak_cl = stage2_best_fitness
        
        # Store the Stage 2 final airfoil for comparison
        self.stage2_final_airfoil = copy.deepcopy(best_individual)
        self._apply_mode_to_ind(self.stage2_final_airfoil)
        
        print(f"\n" + "="*60)
        print(f"STAGE 2 COMPLETE: Peak CL = {self.stage2_best_peak_cl:.3f}")
        print(f"Switching to Stage 3: Optimizing for average CL over ±4° range around peak L/D")
        print(f"Must maintain peak L/D ≥ {self.stage1_best_peak_ld * self.peak_ld_preservation_threshold:.3f} ({self.peak_ld_preservation_threshold*100:.0f}% of best)")
        print(f"Must maintain peak CL ≥ {self.stage2_best_peak_cl * self.peak_cl_preservation_threshold:.3f} ({self.peak_cl_preservation_threshold*100:.0f}% of best)")
        print(f"Stage 3 focus: Maximize average CL over ±4° range around peak L/D")
        print(f"="*60)
        
        # Stage 3: Optimize for average CL over ±4° range around peak L/D
        print(f"\nStage 3: Optimizing for average CL over ±4° range (preserving peaks)")
        print(f"Additional generations: {self.stage3_generations}")
        print(f"Range optimization: Maximize average CL over ±4° around peak L/D")
        print(f"Stage 3 early stopping patience: 50 generations")
        
        # Reset early stopping for stage 3 with 50 generation patience
        generations_without_improvement = 0
        stage3_best_fitness = stage2_best_fitness
        stage3_early_stopping_patience = 50  # Fixed 50 generation patience for stage 3
        
        for generation in range(self.stage3_generations):
            print(f"\nStage 3 Generation {generation + 1}/{self.stage3_generations}")
            
            self.eval_population()
            
            fitnesses = [ind['fitness'] for ind in self.population]
            best_fitness = float(max(fitnesses))
            avg_fitness = float(np.mean(fitnesses))
            best_individual = max(self.population, key=lambda x: x['fitness'])
            
            # Check if we're still improving in stage 3
            if best_fitness > stage3_best_fitness + self.early_stopping_tolerance:
                stage3_best_fitness = best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            print(f"Best fitness: {best_fitness:.3f}")
            print(f"Average fitness: {avg_fitness:.3f}")
            print(f"Generations without improvement: {generations_without_improvement}/{stage3_early_stopping_patience}")
            
            # Early stopping for stage 3 with 50 generation patience
            if generations_without_improvement >= stage3_early_stopping_patience:
                print(f"Stage 3 early stopping triggered! No improvement for {stage3_early_stopping_patience} generations.")
                break
            
            # Add to fitness history with stage indicator
            self.fitness_history.append({
                'generation': len(self.fitness_history) + 1,
                'stage': 3,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_individual': copy.deepcopy(best_individual)
            })
            
            if self.image_callback:
                try:
                    self.image_callback(best_individual, len(self.fitness_history), best_fitness)
                except Exception as e:
                    print(f"Image capture failed for stage 3 generation {generation + 1}: {e}")
            
            if generation < self.stage3_generations - 1:
                # Create next generation WITHOUT adding noise (clean optimization)
                self.next_generation_stage2()
        
        # Stage 3 complete - record best average CL
        self.stage3_complete = True
        self.stage3_best_avg_cl = stage3_best_fitness
        
        # Store the Stage 3 final airfoil for comparison
        self.stage3_final_airfoil = copy.deepcopy(best_individual)
        self._apply_mode_to_ind(self.stage3_final_airfoil)
        
        print(f"\n" + "="*60)
        print(f"STAGE 3 COMPLETE: Average CL = {self.stage3_best_avg_cl:.3f}")
        print(f"Switching to Stage 4: Optimizing for average L/D over ±4° range around peak L/D")
        print(f"Must maintain peak L/D ≥ {self.stage1_best_peak_ld * self.peak_ld_preservation_threshold:.3f} ({self.peak_ld_preservation_threshold*100:.0f}% of best)")
        print(f"Must maintain peak CL ≥ {self.stage2_best_peak_cl * self.peak_cl_preservation_threshold:.3f} ({self.peak_cl_preservation_threshold*100:.0f}% of best)")
        print(f"Stage 4 focus: Maximize average L/D over ±4° range around peak L/D")
        print(f"="*60)
        
        # Stage 4: Optimize for average L/D over ±4° range around peak L/D
        print(f"\nStage 4: Optimizing for average L/D over ±4° range (preserving peaks)")
        print(f"Additional generations: {self.stage4_generations}")
        print(f"Range optimization: Maximize average L/D over ±4° around peak L/D")
        print(f"Stage 4 early stopping patience: 50 generations")
        
        # Reset early stopping for stage 4 with 50 generation patience
        generations_without_improvement = 0
        stage4_best_fitness = stage3_best_fitness
        stage4_early_stopping_patience = 50  # Fixed 50 generation patience for stage 4
        
        for generation in range(self.stage4_generations):
            print(f"\nStage 4 Generation {generation + 1}/{self.stage4_generations}")
            
            self.eval_population()
            
            fitnesses = [ind['fitness'] for ind in self.population]
            best_fitness = float(max(fitnesses))
            avg_fitness = float(np.mean(fitnesses))
            best_individual = max(self.population, key=lambda x: x['fitness'])
            
            # Check if we're still improving in stage 4
            if best_fitness > stage4_best_fitness + self.early_stopping_tolerance:
                stage4_best_fitness = best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            print(f"Best fitness: {best_fitness:.3f}")
            print(f"Average fitness: {avg_fitness:.3f}")
            print(f"Generations without improvement: {generations_without_improvement}/{stage4_early_stopping_patience}")
            
            # Early stopping for stage 4 with 50 generation patience
            if generations_without_improvement >= stage4_early_stopping_patience:
                print(f"Stage 4 early stopping triggered! No improvement for {stage4_early_stopping_patience} generations.")
                break
            
            # Add to fitness history with stage indicator
            self.fitness_history.append({
                'generation': len(self.fitness_history) + 1,
                'stage': 4,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_individual': copy.deepcopy(best_individual)
            })
            
            if self.image_callback:
                try:
                    self.image_callback(best_individual, len(self.fitness_history), best_fitness)
                except Exception as e:
                    print(f"Image capture failed for stage 4 generation {generation + 1}: {e}")
            
            if generation < self.stage4_generations - 1:
                # Create next generation WITHOUT adding noise (clean optimization)
                self.next_generation_stage2()
        
        # Final evaluation
        self.eval_population()
        final_best = max(self.population, key=lambda x: x['fitness'])
        self._apply_mode_to_ind(final_best)
        
        print(f"\nOptimization Complete!")
        print(f"Stage 1: Peak L/D = {self.stage1_best_peak_ld:.3f}")
        print(f"Stage 2: Peak CL = {self.stage2_best_peak_cl:.3f}")
        print(f"Stage 3: Average CL = {self.stage3_best_avg_cl:.3f}")
        print(f"Stage 4: Final fitness = {final_best['fitness']:.3f}")
        
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
    
    def plot_history(self):
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
    
    def stage1_airfoil(self):
        """Get the final airfoil from Stage 1 (non-area optimized) for comparison"""
        return self.stage1_final_airfoil
    
    def early_stop_stats(self):
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
            'stage1_final_airfoil': self.stage1_airfoil()
        }
    
    def save_report(self, filename: str = 'af512_optimization_results.json'):
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
    
    best_airfoil = optimizer.run(initial_naca)
    
    optimizer.plot_history()
    
    optimizer.save_report()
    
    if initial_naca:
        print(f"\nComparison:")
        initial_individual = optimizer.make_from_naca(initial_naca)
        initial_fitness = optimizer.eval_fitness(initial_individual)
        print(f"Initial NACA {initial_naca}: L/D = {initial_fitness:.3f}")
        print(f"Optimized AF512: L/D = {best_airfoil['fitness']:.3f}")
        improvement = ((best_airfoil['fitness'] - initial_fitness) / initial_fitness) * 100
        print(f"Improvement: {improvement:.1f}%")

if __name__ == "__main__":
    main()

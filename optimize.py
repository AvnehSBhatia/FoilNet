#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from genetic_naca_optimizer import GeneticAF512Optimizer
import neuralfoil
import json
import os
import cv2
from PIL import Image
import io
from tqdm import tqdm

#EDIT THESE VALUES TO CHANGE THE FLIGHT CONDITIONS
WINGSPAN = 5
CHORD = 0.8
AIRSPEED = 35

def optimize_naca_and_compare():
    print("Multi-NACA Airfoil Optimizer")
    print("=" * 40)
    print(f"Flight Conditions: {WINGSPAN}ft wingspan, {CHORD}ft chord, {AIRSPEED}mph airspeed")
    print("=" * 40)
    
    target_lift = None
    try:
        pounds_input = input("Target lift (pounds): ").strip()
        if pounds_input:
            pounds = float(pounds_input)
            rho = 0.002377
            velocity = AIRSPEED * 1.467
            wing_area = WINGSPAN * CHORD
            target_lift = pounds / (0.5 * rho * velocity**2 * wing_area)
            print(f"Target lift: {pounds} lbs → CL = {target_lift:.3f}")
    except ValueError:
        print("Invalid target lift. Proceeding without lift constraint.")
    
    try:
        pop_size = int(input("Population size (default 5): ") or "5")
        gens = int(input("Generations (default 1000): ") or "1000")
        early_stop_patience = int(input("Early stopping patience (default 50): ") or "50")
        early_stop_tolerance = float(input("Early stopping tolerance (default 0.2): ") or "0.2")
    except ValueError:
        pop_size, gens = 30, 1000
        early_stop_patience, early_stop_tolerance = 50, 0.001
        print("Using default values")
    
    optimizer = GeneticAF512Optimizer(
        population_size=pop_size,
        generations=gens,
        wingspan=WINGSPAN,
        chord=CHORD,
        airspeed=AIRSPEED,
        mutation_rate=0.15,
        crossover_rate=0.8,
        target_lift=target_lift,
        early_stopping_patience=early_stop_patience,
        early_stopping_tolerance=early_stop_tolerance
    )
    
    # Get user input for base NACA profile (NACA format only)
    base_naca = None
    try:
        naca_input = input("Enter base NACA profile (e.g., 2412) or press Enter for ellipse: ").strip()
        if naca_input:
            # Validate that it's a NACA format (4 digits)
            if naca_input.isdigit() and len(naca_input) == 4:
                base_naca = naca_input
                print(f"Using NACA {base_naca} as starting profile")
            else:
                print(f"Invalid NACA format. {naca_input} is not a valid 4-digit NACA code.")
                print("Using ellipse-based starting shape instead.")
                base_naca = None
        else:
            print("Using ellipse-based starting shape")
    except:
        print("Using ellipse-based starting shape")
    
    print(f"\nFour-Stage Optimization: {'NACA profile' if base_naca else 'ellipse-based starting shapes'}")
    print(f"Stage 1: Peak L/D optimization ({gens} generations with early stopping patience {early_stop_patience})")
    print(f"Stage 2: Peak CL at peak L/D optimization (up to 1000 additional generations)")
    print(f"Stage 3: Average CL over ±4° range optimization (up to 1000 additional generations)")
    print(f"Stage 4: Average L/D over ±4° range optimization (up to 1000 additional generations)")
    print(f"Constraints: Peak L/D and peak CL must not drop more than 2% from their respective stage bests")
    
    def create_ellipse_af512(thickness=0.12, thickness_pos=0.5, num_points=512):
        x_points = np.linspace(0, 1, num_points)
        
        a = 0.5
        b = thickness / 2
        
        thickness_dist = np.zeros(num_points)
        for i, x in enumerate(x_points):
            dx = (x - thickness_pos) / a
            if abs(dx) <= 1:
                thickness_dist[i] = b * np.sqrt(1 - dx**2)
            else:
                thickness_dist[i] = 0
        
        upper_dist = thickness_dist
        lower_dist = -thickness_dist
        
        upper_dist[0] = 0.0
        lower_dist[0] = 0.0
        
        return np.column_stack([upper_dist, lower_dist])
    
    starting_shapes = []
    
    if base_naca:
        # Use user-specified NACA profile
        starting_shapes.append((f'NACA {base_naca}', None))
    else:
        # Use default ellipse-based shape
        starting_shapes.append(('Ellipse (12% thickness, centered)', create_ellipse_af512(0.12, 0.5)))
    
    naca_results = []
    
    for i, (shape_name, af512_data) in enumerate(starting_shapes):
        print(f"\n   Optimizing {shape_name} ({i+1}/{len(starting_shapes)})...")
        
        image_folder = f"optimization_images_{i+1}"
        os.makedirs(image_folder, exist_ok=True)
        
        def capture_generation_image(best_individual, generation, fitness):
            image_path = os.path.join(image_folder, f"gen_{generation:03d}_L{fitness:.1f}.png")
            capture_airfoil_image(best_individual, generation, fitness, image_path)
        
        if af512_data is not None:
            original_individual = {
                'af512_data': af512_data,
                'fitness': None,
                'xy_coordinates': None,
                'aerodynamic_data': None
            }
        else:
            # Extract airfoil code from shape name for optimization
            airfoil_code = shape_name  # The shape_name is now just the airfoil code
            original_individual = optimizer.create_individual_from_airfoil(airfoil_code)
        
        original_fitness = optimizer.evaluate_fitness(original_individual)
        
        temp_optimizer = GeneticAF512Optimizer(
            population_size=pop_size,
            generations=gens,
            wingspan=WINGSPAN,
            chord=CHORD,
            airspeed=AIRSPEED,
            mutation_rate=0.15,
            crossover_rate=0.8,
            target_lift=target_lift,
            image_callback=capture_generation_image,
            early_stopping_patience=early_stop_patience,
            early_stopping_tolerance=early_stop_tolerance
        )
        
        if af512_data is not None:
            temp_optimizer.initialize_population()
            temp_optimizer.population[0] = {
                'af512_data': af512_data.copy(),
                'fitness': None,
                'xy_coordinates': None,
                'aerodynamic_data': None
            }
            optimized_airfoil = temp_optimizer.run_optimization()
        else:
            # Extract airfoil code from shape name for optimization
            airfoil_code = shape_name  # The shape_name is now just the airfoil code
            optimized_airfoil = temp_optimizer.run_optimization(airfoil_code)
        
        naca_results.append({
            'naca': shape_name,
            'original_airfoil': original_individual,
            'original_fitness': original_fitness,
            'optimized_airfoil': optimized_airfoil,
            'fitness': optimized_airfoil['fitness'],
            'optimizer': temp_optimizer
        })
        
        print(f"   {shape_name}: Original Peak L/D = {original_fitness:.3f} → Final Stage 4 Fitness = {optimized_airfoil['fitness']:.3f}")
        print(f"   Stage 1 Peak L/D: {optimized_airfoil['aerodynamic_data']['L/D']:.3f}")
        print(f"   Stage 2 Peak CL: {optimized_airfoil['aerodynamic_data']['CL']:.3f}")
        print(f"   Stage 4 Final Fitness: {optimized_airfoil['fitness']:.3f}")
    
    best_naca_result = max(naca_results, key=lambda x: x['fitness'])
    best_shape_name = best_naca_result['naca']
    best_optimized_airfoil = best_naca_result['optimized_airfoil']
    
    print(f"\nBest optimized shape: {best_shape_name}")
    print(f"  Stage 1 Peak L/D: {best_optimized_airfoil['aerodynamic_data']['L/D']:.3f}")
    print(f"  Stage 2 Peak CL: {best_optimized_airfoil['aerodynamic_data']['CL']:.3f}")
    print(f"  Stage 4 Final Fitness: {best_naca_result['fitness']:.3f}")
    
    print(f"\n" + "="*50)
    print(f"EARLY STOPPING STATISTICS")
    print(f"="*50)
    early_stopped_count = 0
    total_generations_saved = 0
    
    for i, result in enumerate(naca_results):
        if 'optimizer' in result:
            stats = result['optimizer'].get_early_stopping_stats()
            if stats:
                print(f"{result['naca']}:")
                print(f"  - Completed {stats['total_generations']}/{stats['max_generations']} generations")
                print(f"  - Early stopped: {'Yes' if stats['early_stopped'] else 'No'}")
                print(f"  - Best Stage 4 Final Fitness: {stats['best_fitness']:.3f} at generation {stats['best_fitness_generation']}")
                
                if stats['stage1_complete']:
                    print(f"  - Stage 1: {stats['stage1_generations']} gens, Peak L/D: {stats['stage1_best_peak_ld']:.3f}")
                    print(f"  - Stage 2: {stats['stage2_generations']} gens, Best fitness: {stats['stage2_best_fitness']:.3f}")
                    if stats['peak_ld_preserved']:
                        print(f"  - Peak L/D preserved within 98% threshold")
                    else:
                        print(f"  - Peak L/D not preserved within 98% threshold")
                
                if stats['early_stopped']:
                    early_stopped_count += 1
                    total_generations_saved += (stats['max_generations'] - stats['total_generations'])
    
    if early_stopped_count > 0:
        print(f"\nEarly Stopping Summary:")
        print(f"  - {early_stopped_count}/{len(naca_results)} optimizations used early stopping")
        print(f"  - Total generations saved: {total_generations_saved}")
        print(f"  - Average generations saved per optimization: {total_generations_saved/early_stopped_count:.1f}")
    else:
        print(f"\nNo early stopping occurred - all optimizations ran to completion")
    
    print(f"\n" + "="*50)
    print(f"FOUR-STAGE OPTIMIZATION RESULTS")
    print(f"="*50)
    
    for i, result in enumerate(naca_results):
        if 'optimizer' in result:
            stats = result['optimizer'].get_early_stopping_stats()
            if stats and stats['stage1_complete']:
                print(f"{result['naca']}:")
                print(f"  Stage 1 Peak L/D: {stats['stage1_best_peak_ld']:.3f}")
                print(f"  Stage 2 Peak CL: {stats.get('stage2_best_peak_cl', 0):.3f}")
                print(f"  Stage 3 Average CL: {stats.get('stage3_best_avg_cl', 0):.3f}")
                print(f"  Stage 4 Final Fitness: {result['fitness']:.3f}")
                print(f"  Final Peak L/D: {result['optimized_airfoil']['aerodynamic_data']['L/D']:.3f}")
                print(f"  Final Peak CL: {result['optimized_airfoil']['aerodynamic_data']['CL']:.3f}")
                print(f"  Peak Alpha: {result['optimized_airfoil']['aerodynamic_data']['peak_alpha']:.1f}°")
                
                # Calculate improvement in area under curve
                original_area = result['original_airfoil']['aerodynamic_data'].get('area_under_curve', 0)
                final_area = result['optimized_airfoil']['aerodynamic_data'].get('area_under_curve', 0)
                if original_area > 0:
                    area_improvement = ((final_area - original_area) / original_area) * 100
                    print(f"  Area under curve improvement: {area_improvement:+.1f}%")
                
                # Check L/D falloff characteristics
                if 'ld_values' in result['optimized_airfoil']['aerodynamic_data']:
                    ld_values = result['optimized_airfoil']['aerodynamic_data']['ld_values']
                    if len(ld_values) >= 21:  # Full alpha range
                        peak_ld = max(ld_values)
                        ld_at_10deg = ld_values[15]  # Index 15 = 10 degrees
                        if peak_ld > 0:
                            falloff_ratio = ld_at_10deg / peak_ld
                            print(f"  L/D at 10°: {ld_at_10deg:.2f} ({falloff_ratio*100:.1f}% of peak)")
                            if falloff_ratio >= 0.9:
                                print(f"  L/D maintained within 10% of peak at 10°")
                            else:
                                print(f"  L/D drops below 90% of peak at 10°")
                
                print()
    
    print(f"\n" + "="*50)
    print(f"OPTIMIZATION RESULTS")
    print(f"="*50)
    print(f"Best optimized shape: {best_shape_name}")
    print(f"  Stage 1 Peak L/D: {best_optimized_airfoil['aerodynamic_data']['L/D']:.3f}")
    print(f"  Stage 2 Peak CL: {best_optimized_airfoil['aerodynamic_data']['CL']:.3f}")
    print(f"  Stage 4 Final Fitness: {best_naca_result['fitness']:.3f}")
    
    results_data = {
        'target_lift': target_lift,
        'target_lift_lbs': pounds if target_lift else None,
        'flight_conditions': {
            'wingspan': WINGSPAN,
            'chord': CHORD,
            'airspeed': AIRSPEED
        },
        'optimization_parameters': {
            'population_size': pop_size,
            'generations': gens,
            'early_stopping_patience': early_stop_patience,
            'early_stopping_tolerance': early_stop_tolerance
        },
        'all_naca_results': [
            {
                'naca': result['naca'],
                'fitness': result['fitness'],
                'peak_alpha': result['optimized_airfoil']['aerodynamic_data']['peak_alpha'],
                'cl': result['optimized_airfoil']['aerodynamic_data']['CL'],
                'cd': result['optimized_airfoil']['aerodynamic_data']['CD']
            }
            for result in naca_results
        ],
        'best_shape': {
            'name': best_shape_name,
            'fitness': best_naca_result['fitness'],
            'peak_alpha': best_optimized_airfoil['aerodynamic_data']['peak_alpha'],
            'cl': best_optimized_airfoil['aerodynamic_data']['CL'],
            'cd': best_optimized_airfoil['aerodynamic_data']['CD']
        }
    }
    
    save_dat_file_only(best_optimized_airfoil, best_shape_name, optimizer)
    
    best_image_folder = f"optimization_images_{naca_results.index(best_naca_result) + 1}"
    video_path = f"optimization_video_{best_shape_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')}.mp4"
    create_optimization_video(best_image_folder, video_path, target_duration=10.0)
    
    create_comprehensive_plots(naca_results, best_naca_result, optimizer, target_lift, early_stop_patience, early_stop_tolerance)
    
    return best_optimized_airfoil, naca_results

def create_comparison_plots(original, optimized, optimizer, naca_input):
    print(f"\nCreating comparison plots...")
    
    print(f"Original keys: {list(original.keys())}")
    print(f"Optimized keys: {list(optimized.keys())}")
    print(f"Original has af512_data: {'af512_data' in original}")
    print(f"Optimized has af512_data: {'af512_data' in optimized}")
    
    orig_x, orig_y = original['xy_coordinates']
    opt_x, opt_y = optimized['xy_coordinates']
    
    fig = plt.figure(figsize=(16, 10))
    
    fig.suptitle(f'Flight Conditions: {optimizer.wingspan}ft wingspan, {optimizer.chord}ft chord, {optimizer.airspeed}mph airspeed (Re ≈ {optimizer.reynolds_number:.0f}) | Four-Stage: Peak L/D → Peak CL → Avg CL (±4°) → Avg L/D (±4°)', 
                 fontsize=14, fontweight='bold')
    
    plt.subplot(2, 3, 1)
    plt.plot(orig_x, orig_y, 'b-', linewidth=3, label=f'NACA {naca_input}', alpha=0.8)
    plt.plot(opt_x, opt_y, 'r-', linewidth=3, label='Optimized AF512', alpha=0.8)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Airfoil Shape Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.subplot(2, 3, 2)
    labels = ['Original', 'Optimized']
    ld_ratios = [original['fitness'], optimized['fitness']]
    colors = ['blue', 'red']
    
    bars = plt.bar(labels, ld_ratios, color=colors, alpha=0.7)
    plt.ylabel('Stage 4 Final Fitness')
    plt.title('Stage 4 Performance Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, ld_ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.subplot(2, 3, 3)
    orig_cl = original['aerodynamic_data']['CL']
    orig_cd = original['aerodynamic_data']['CD']
    opt_cl = optimized['aerodynamic_data']['CL']
    opt_cd = optimized['aerodynamic_data']['CD']
    
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, [orig_cl, opt_cl], width, label='CL', alpha=0.7, color='green')
    plt.bar(x + width/2, [orig_cd, opt_cd], width, label='CD', alpha=0.7, color='orange')
    plt.xlabel('Airfoil')
    plt.ylabel('Coefficient')
    plt.title('Aerodynamic Coefficients')
    plt.xticks(x, ['Original', 'Optimized'])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(2, 3, 4)
    x_points = np.linspace(0, 1, 512)
    
    orig_af512 = original['af512_data']
    plt.plot(x_points, orig_af512[:, 0], 'b-', linewidth=2, label='Original Upper', alpha=0.7)
    plt.plot(x_points, orig_af512[:, 1], 'b--', linewidth=2, label='Original Lower', alpha=0.7)
    
    opt_af512 = optimized['af512_data']
    plt.plot(x_points, opt_af512[:, 0], 'r-', linewidth=2, label='Optimized Upper', alpha=0.7)
    plt.plot(x_points, opt_af512[:, 1], 'r--', linewidth=2, label='Optimized Lower', alpha=0.7)
    
    plt.xlabel('Chord Position (0-1)')
    plt.ylabel('Distance from Chord Line')
    plt.title('AF512 Format Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if hasattr(optimizer, 'fitness_history') and optimizer.fitness_history:
        plt.subplot(2, 3, 5)
        generations = [h['generation'] for h in optimizer.fitness_history]
        best_fitnesses = [h['best_fitness'] for h in optimizer.fitness_history]
        avg_fitnesses = [h['avg_fitness'] for h in optimizer.fitness_history]
        
        plt.plot(generations, best_fitnesses, 'b-', linewidth=2, label='Best Fitness', marker='o')
        plt.plot(generations, avg_fitnesses, 'r--', linewidth=2, label='Average Fitness', alpha=0.7)
        plt.axhline(y=original['fitness'], color='g', linestyle=':', linewidth=2, label='Original Peak L/D')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title('Four-Stage Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    
    alpha_range = np.linspace(-5, 15, 21)
    
    original_ld_values = []
    optimized_ld_values = []
    
    print(f"Calculating L/D vs Alpha for comparison...")
    
    for alpha in alpha_range:
        try:
            orig_x, orig_y = original['xy_coordinates']
            orig_coords = np.column_stack([orig_x, orig_y])
            orig_results = neuralfoil.get_aero_from_coordinates(
                coordinates=orig_coords,
                alpha=alpha,
                Re=optimizer.reynolds_number,
                model_size="xxxlarge"
            )
            orig_cl = float(orig_results['CL'].item() if hasattr(orig_results['CL'], 'item') else orig_results['CL'])
            orig_cd = float(orig_results['CD'].item() if hasattr(orig_results['CD'], 'item') else orig_results['CD'])
            orig_ld = orig_cl / orig_cd if orig_cd > 0 else 0
            original_ld_values.append(orig_ld)
            
            opt_x, opt_y = optimized['xy_coordinates']
            opt_coords = np.column_stack([opt_x, opt_y])
            opt_results = neuralfoil.get_aero_from_coordinates(
                coordinates=opt_coords,
                alpha=alpha,
                Re=optimizer.reynolds_number,
                model_size="xxxlarge"
            )
            opt_cl = float(opt_results['CL'].item() if hasattr(opt_results['CL'], 'item') else opt_results['CL'])
            opt_cd = float(opt_results['CD'].item() if hasattr(opt_results['CD'], 'item') else opt_results['CD'])
            opt_ld = opt_cl / opt_cd if opt_cd > 0 else 0
            optimized_ld_values.append(opt_ld)
            
        except Exception as e:
            print(f"Error at alpha {alpha}°: {e}")
            original_ld_values.append(0)
            optimized_ld_values.append(0)
    
    plt.plot(alpha_range, original_ld_values, 'b-', linewidth=2, label=f'NACA {naca_input}', alpha=0.8)
    plt.plot(alpha_range, optimized_ld_values, 'r-', linewidth=2, label='Optimized AF512', alpha=0.8)
    
    orig_peak_idx = np.argmax(original_ld_values)
    opt_peak_idx = np.argmax(optimized_ld_values)
    
    plt.plot(alpha_range[orig_peak_idx], original_ld_values[orig_peak_idx], 'bo', markersize=8, label=f'Original Peak ({alpha_range[orig_peak_idx]:.1f}°)')
    plt.plot(alpha_range[opt_peak_idx], optimized_ld_values[opt_peak_idx], 'ro', markersize=8, label=f'Optimized Peak ({alpha_range[opt_peak_idx]:.1f}°)')
    
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('L/D Ratio')
    plt.title('L/D vs Angle of Attack')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Comparison displayed")
    
    return original_ld_values, optimized_ld_values, alpha_range

def convert_dat_to_dxf(dat_filename, dxf_filename, shape_name, fitness):
    print(f"   Converting DAT to DXF with smooth splines...")
    
    with open(dat_filename, 'r') as f:
        lines = f.readlines()
    
    coord_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('Optimized') and not line.startswith('L/D') and not line.startswith('CL') and not line.startswith('Flight') and not line.startswith('Reynolds'):
            try:
                parts = line.split()
                if len(parts) == 2:
                    x, y = float(parts[0]), float(parts[1])
                    coord_lines.append((x, y))
            except ValueError:
                continue
    
    if not coord_lines:
        print(f"   No coordinates found in DAT file")
        return
    
    coords = np.array(coord_lines)
    
    scale = 152.4
    scaled_coords = coords * scale
    
    print(f"   Scaling coordinates by {scale} mm (0.5 ft chord)")
    print(f"   Original coordinates range: X[{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}], Y[{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
    print(f"   Scaled coordinates range: X[{scaled_coords[:, 0].min():.1f}, {scaled_coords[:, 0].max():.1f}] mm, Y[{scaled_coords[:, 1].min():.1f}, {scaled_coords[:, 1].max():.1f}] mm")
    
    mid_point = len(scaled_coords) // 2
    
    upper_surface = scaled_coords[:mid_point+1]
    
    lower_surface = np.vstack([scaled_coords[0], scaled_coords[mid_point:], scaled_coords[0]])
    
    print(f"   Creating DXF with two splines:")
    print(f"      Upper surface: {len(upper_surface)} points")
    print(f"      Lower surface: {len(lower_surface)} points")
    
    with open(dxf_filename, "w") as f:
        f.write("0\nSECTION\n2\nHEADER\n0\nENDSEC\n")
        f.write("0\nSECTION\n2\nTABLES\n0\nENDSEC\n")
        f.write("0\nSECTION\n2\nBLOCKS\n0\nENDSEC\n")
        f.write("0\nSECTION\n2\nENTITIES\n")
        
        f.write("0\nLWPOLYLINE\n8\n0\n90\n{}\n70\n0\n".format(len(upper_surface)))
        for x, y in upper_surface:
            f.write("10\n{:.6f}\n20\n{:.6f}\n".format(x, y))
        
        f.write("0\nLWPOLYLINE\n8\n0\n90\n{}\n70\n0\n".format(len(lower_surface)))
        for x, y in lower_surface:
            f.write("10\n{:.6f}\n20\n{:.6f}\n".format(x, y))
        
        f.write("0\nENDSEC\n0\nEOF\n")
    
    print(f"   DXF file saved as: {dxf_filename}")
    print(f"   DXF conversion complete - two smooth splines ready for Onshape import")

def capture_airfoil_image(airfoil_data, generation, fitness, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_coords, y_coords = airfoil_data['xy_coordinates']
    
    ax.plot(x_coords, y_coords, 'b-', linewidth=3, alpha=0.8)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Generation {generation} - L/D = {fitness:.3f}')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 0.2)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    
    img = Image.open(buf)
    img.save(save_path)
    plt.close(fig)
    buf.close()
    
    return save_path

def create_optimization_video(image_folder, output_video_path, target_duration=10.0):
    print(f"Creating optimization video...")
    
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not image_files:
        print("No images found for video creation")
        return
    
    print(f"Found {len(image_files)} images for video creation")
    
    fps = len(image_files) / target_duration
    
    fps = max(5, fps)
    
    actual_duration = len(image_files) / fps
    
    print(f"Target duration: {target_duration}s, Calculated FPS: {fps:.1f}")
    
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        video.write(frame)
    
    video.release()
    print(f"Video saved as: {output_video_path}")
    print(f"Video stats: {len(image_files)} generations, {fps:.1f} fps, {actual_duration:.1f}s duration")
    
    try:
        keep_images = input("Keep individual generation images? (y/n, default n): ").strip().lower()
        if keep_images not in ['y', 'yes']:
            print("Cleaning up individual images...")
            for image_file in image_files:
                os.remove(os.path.join(image_folder, image_file))
            os.rmdir(image_folder)
            print("Images cleaned up")
        else:
            print("Individual images kept in 'optimization_images' folder")
    except:
        print("Individual images kept in 'optimization_images' folder")

def save_dat_file_only(best_optimized_airfoil, best_shape_name, optimizer):
    print(f"\nSaving DAT file and converting to DXF...")
    
    x_coords, y_coords = best_optimized_airfoil['xy_coordinates']
    fitness = best_optimized_airfoil['fitness']
    aero_data = best_optimized_airfoil['aerodynamic_data']
    
    # Reduce to 999 points by removing points with lowest concavity
    
    # Extract base NACA code from shape name for AF format
    if ' ' in best_shape_name:
        base_code = best_shape_name.split()[-1]  # Extract the airfoil code
    else:
        base_code = best_shape_name
    
    # Create AF filename (AF + base code)
    filename_base = f"AF{int(round(optimizer.reynolds_number / 1000, 0))}k"
    
    dat_filename = f"{filename_base}.dat"
    with open(dat_filename, 'w') as f:
        f.write(f"{filename_base} Airfoil\n")
        f.write(f"Peak L/D = {aero_data['CL']/aero_data['CD']:.3f} \n")
        f.write(f"Peak CL = {aero_data['CL']:.4f}\n")
        f.write(f"Peak CD = {aero_data['CD']:.4f}\n")
        f.write(f"Peak Angle = {aero_data['peak_alpha']:.2f}°\n")
        f.write(f"Reynolds Number = {optimizer.reynolds_number:.0f}\n")
        f.write(f"1024\n")
        
        for i in range(len(x_coords)):
            f.write(f"{x_coords[i]:.6f} {y_coords[i]:.6f}\n")
    
    print(f"   DAT file saved as: {dat_filename}")    
    dxf_filename = f"{filename_base}.dxf"
    convert_dat_to_dxf(dat_filename, dxf_filename, best_shape_name, fitness)
    
    print(f"DAT and DXF files saved successfully!")

    
    # Calculate concavity (second derivative approximation) for each point
    concavities = []
    for i in range(1, len(x_coords) - 1):
        # Second derivative approximation using finite differences
        h = x_coords[i+1] - x_coords[i-1]
        if h > 0:
            d2y_dx2 = (y_coords[i+1] - 2*y_coords[i] + y_coords[i-1]) / (h**2)
        else:
            d2y_dx2 = 0
        concavities.append(abs(d2y_dx2))
    
    # Add concavity for first and last points (use nearest neighbor)
    concavities.insert(0, concavities[0] if concavities else 0)
    concavities.append(concavities[-1] if concavities else 0)
    
    # Create list of (index, concavity) pairs
    point_concavities = list(enumerate(concavities))
    
    # Sort by concavity (lowest first)
    point_concavities.sort(key=lambda x: x[1])
    
    # Keep the 999 points with highest concavity
    indices_to_keep = [i for i, _ in point_concavities[-(999):]]
    indices_to_keep.sort()  # Sort to maintain order
    
    x_coords_999 = x_coords[indices_to_keep]
    y_coords_999 = y_coords[indices_to_keep]
    
    return x_coords_999, y_coords_999

def create_comprehensive_plots(naca_results, best_naca_result, optimizer, target_lift, early_stop_patience=None, early_stop_tolerance=None):
    print(f"\nCreating comprehensive comparison plots...")
    
    shape_names = [result['naca'] for result in naca_results]
    original_fitness_values = [result['original_fitness'] for result in naca_results]
    optimized_fitness_values = [result['fitness'] for result in naca_results]
    peak_alphas = [result['optimized_airfoil']['aerodynamic_data']['peak_alpha'] for result in naca_results]
    cl_values = [result['optimized_airfoil']['aerodynamic_data']['CL'] for result in naca_results]
    cd_values = [result['optimized_airfoil']['aerodynamic_data']['CD'] for result in naca_results]
    
    # Extract two-stage optimization data
    stage1_peak_ld_values = []
    stage2_final_fitness_values = []
    area_improvements = []
    
    for result in naca_results:
        if 'optimizer' in result:
            stats = result['optimizer'].get_early_stopping_stats()
            if stats and stats['stage1_complete']:
                if stats['stage1_best_peak_ld'] > 0:
                    stage1_peak_ld_values.append(stats['stage1_best_peak_ld'])
                    stage2_final_fitness_values.append(stats['stage2_best_fitness'])
                
                # Calculate area under curve improvement
                original_area = result['original_airfoil']['aerodynamic_data'].get('area_under_curve', 0)
                final_area = result['optimized_airfoil']['aerodynamic_data'].get('area_under_curve', 0)
                if original_area > 0:
                    area_improvement = ((final_area - original_area) / original_area) * 100
                else:
                    area_improvement = 0
                area_improvements.append(area_improvement)
            else:
                stage1_peak_ld_values.append(0)
                stage2_final_fitness_values.append(0)
                area_improvements.append(0)
        else:
            stage1_peak_ld_values.append(0)
            stage2_final_fitness_values.append(0)
            area_improvements.append(0)
    
    alpha_range = np.linspace(-5, 15, 21)
    original_peak_ld_values = []
    optimized_peak_ld_values = []
    
    for result in naca_results:
        orig_x, orig_y = result['original_airfoil']['xy_coordinates']
        orig_coords = np.column_stack([orig_x, orig_y])
        orig_ld_values = []
        
        for alpha in alpha_range:
            try:
                results = neuralfoil.get_aero_from_coordinates(
                    coordinates=orig_coords,
                    alpha=alpha,
                    Re=optimizer.reynolds_number,
                    model_size="xxxlarge"
                )
                cl = float(results['CL'].item() if hasattr(results['CL'], 'item') else results['CL'])
                cd = float(results['CD'].item() if hasattr(results['CD'], 'item') else results['CD'])
                ld = cl / cd if cd > 0 else 0
                orig_ld_values.append(ld)
            except:
                orig_ld_values.append(0)
        
        original_peak_ld_values.append(max(orig_ld_values))
        
        opt_x, opt_y = result['optimized_airfoil']['xy_coordinates']
        opt_coords = np.column_stack([opt_x, opt_y])
        opt_ld_values = []
        
        for alpha in alpha_range:
            try:
                results = neuralfoil.get_aero_from_coordinates(
                    coordinates=opt_coords,
                    alpha=alpha,
                    Re=optimizer.reynolds_number,
                    model_size="xxxlarge"
                )
                cl = float(results['CL'].item() if hasattr(results['CL'], 'item') else results['CL'])
                cd = float(results['CD'].item() if hasattr(results['CD'], 'item') else results['CD'])
                ld = cl / cd if cd > 0 else 0
                opt_ld_values.append(ld)
            except:
                opt_ld_values.append(0)
        
        optimized_peak_ld_values.append(max(opt_ld_values))
    
    fig = plt.figure(figsize=(24, 16))
    
    title = f'Multi-NACA Four-Stage Optimization Results | Flight: {optimizer.wingspan}ft wingspan, {optimizer.chord}ft chord, {optimizer.airspeed}mph'
    if target_lift:
        title += f' | Target Lift: {target_lift:.3f} CL'
    if early_stop_patience and early_stop_tolerance:
        title += f' | Early Stop: {early_stop_patience} gens, tol={early_stop_tolerance}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Four-stage optimization comparison
    plt.subplot(3, 4, 1)
    x = np.arange(len(shape_names))
    width = 0.15
    
    stage1_bars = plt.bar(x - 1.5*width, stage1_peak_ld_values, width, label='Stage 1 (Peak L/D)', alpha=0.7, color='lightblue')
    stage2_bars = plt.bar(x - 0.5*width, [result['optimized_airfoil']['aerodynamic_data']['CL'] for result in naca_results], width, label='Stage 2 (Peak CL)', alpha=0.7, color='orange')
    stage3_bars = plt.bar(x + 0.5*width, [stats.get('stage3_best_avg_cl', 0) for stats in [result['optimizer'].get_early_stopping_stats() if 'optimizer' in result else {} for result in naca_results]], width, label='Stage 3 (Avg CL)', alpha=0.7, color='green')
    stage4_bars = plt.bar(x + 1.5*width, optimized_fitness_values, width, label='Stage 4 (Avg L/D)', alpha=0.7, color='purple')
    
    best_idx = shape_names.index(best_naca_result['naca'])
    stage4_bars[best_idx].set_color('darkred')
    
    plt.xlabel('Starting Shape')
    plt.ylabel('Fitness Value')
    plt.title('Four-Stage Optimization Results')
    plt.xticks(x, shape_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Stage 4 Final Fitness performance
    plt.subplot(3, 4, 2)
    colors_area = ['red' if result['naca'] == best_naca_result['naca'] else 'blue' for result in naca_results]
    bars = plt.bar(range(len(shape_names)), optimized_fitness_values, color=colors_area, alpha=0.7)
    plt.xlabel('Starting Shape')
    plt.ylabel('Stage 4 Final Fitness')
    plt.title('Stage 4 Final Fitness Performance')
    plt.xticks(range(len(shape_names)), shape_names, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, optimized_fitness_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Peak L/D comparison
    plt.subplot(3, 4, 3)
    x = np.arange(len(shape_names))
    width = 0.35
    
    original_bars = plt.bar(x - width/2, original_peak_ld_values, width, label='Original Peak', alpha=0.7, color='lightblue')
    optimized_bars = plt.bar(x + width/2, optimized_peak_ld_values, width, label='Final Peak', alpha=0.7, color='orange')
    
    best_idx = shape_names.index(best_naca_result['naca'])
    optimized_bars[best_idx].set_color('red')
    
    plt.xlabel('Starting Shape')
    plt.ylabel('Peak L/D')
    plt.title('Peak L/D Performance: Original vs Final')
    plt.xticks(x, shape_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, (orig_bar, opt_bar) in enumerate(zip(original_bars, optimized_bars)):
        plt.text(orig_bar.get_x() + orig_bar.get_width()/2, orig_bar.get_height() + 0.5, 
                f'{original_peak_ld_values[i]:.1f}', ha='center', va='bottom', fontsize=8)
        plt.text(opt_bar.get_x() + opt_bar.get_width()/2, opt_bar.get_height() + 0.5, 
                f'{optimized_peak_ld_values[i]:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot 4: Optimal angle of attack
    plt.subplot(3, 4, 4)
    colors_alpha = ['red' if result['naca'] == best_naca_result['naca'] else 'blue' for result in naca_results]
    plt.bar(range(len(shape_names)), peak_alphas, color=colors_alpha, alpha=0.7)
    plt.xlabel('Starting Shape')
    plt.ylabel('Peak Alpha (degrees)')
    plt.title('Optimal Angle of Attack')
    plt.xticks(range(len(shape_names)), shape_names, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Lift coefficient vs alpha
    plt.subplot(3, 4, 5)
    alpha_range = np.linspace(-5, 15, 21)
    
    for i, result in enumerate(naca_results):
        opt_x, opt_y = result['optimized_airfoil']['xy_coordinates']
        opt_coords = np.column_stack([opt_x, opt_y])
        
        cl_values_alpha = []
        for alpha in alpha_range:
            try:
                results = neuralfoil.get_aero_from_coordinates(
                    coordinates=opt_coords,
                    alpha=alpha,
                    Re=optimizer.reynolds_number,
                    model_size="xxxlarge"
                )
                cl = float(results['CL'].item() if hasattr(results['CL'], 'item') else results['CL'])
                cl_values_alpha.append(cl)
            except:
                cl_values_alpha.append(0)
        
        color = 'red' if result['naca'] == best_naca_result['naca'] else 'blue'
        plt.plot(alpha_range, cl_values_alpha, color=color, alpha=0.7, linewidth=2)
    
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('Lift Coefficient (CL)')
    plt.title('Lift Coefficient vs Alpha')
    plt.grid(True, alpha=0.3)
    
    if target_lift:
        plt.axhline(y=target_lift, color='red', linestyle='--', alpha=0.7, label=f'Target CL = {target_lift:.3f}')
        plt.legend()
    
    # Plot 6: Drag coefficient vs alpha
    plt.subplot(3, 4, 6)
    
    for i, result in enumerate(naca_results):
        opt_x, opt_y = result['optimized_airfoil']['xy_coordinates']
        opt_coords = np.column_stack([opt_x, opt_y])
        
        cd_values_alpha = []
        for alpha in alpha_range:
            try:
                results = neuralfoil.get_aero_from_coordinates(
                    coordinates=opt_coords,
                    alpha=alpha,
                    Re=optimizer.reynolds_number,
                    model_size="xxxlarge"
                )
                cd = float(results['CD'].item() if hasattr(results['CD'], 'item') else results['CD'])
                cd_values_alpha.append(cd)
            except:
                cd_values_alpha.append(0.1)
        
        color = 'red' if result['naca'] == best_naca_result['naca'] else 'blue'
        plt.plot(alpha_range, cd_values_alpha, color=color, alpha=0.7, linewidth=2)
    
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('Drag Coefficient (CD)')
    plt.title('Drag Coefficient vs Alpha')
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Top 3 airfoil shapes
    plt.subplot(3, 4, 7)
    sorted_results = sorted(naca_results, key=lambda x: x['fitness'], reverse=True)[:3]
    colors_3 = ['red', 'orange', 'green']
    
    for i, result in enumerate(sorted_results):
        result_idx = naca_results.index(result)
        
        orig_x, orig_y = result['original_airfoil']['xy_coordinates']
        plt.plot(orig_x, orig_y, color=colors_3[i], linewidth=1, linestyle='--', 
                label=f"{result['naca']} Original (L/D={original_peak_ld_values[result_idx]:.1f})", alpha=0.6)
        
        # Plot Stage 1 final airfoil (non-area optimized) if available
        if 'optimizer' in result and result['optimizer'].get_stage1_final_airfoil():
            stage1_airfoil = result['optimizer'].get_stage1_final_airfoil()
            if stage1_airfoil and 'xy_coordinates' in stage1_airfoil:
                stage1_x, stage1_y = stage1_airfoil['xy_coordinates']
                plt.plot(stage1_x, stage1_y, color=colors_3[i], linewidth=2, linestyle=':', 
                        label=f"{result['naca']} Stage 1 Final (Peak L/D)", alpha=0.8)
        
        opt_x, opt_y = result['optimized_airfoil']['xy_coordinates']
        plt.plot(opt_x, opt_y, color=colors_3[i], linewidth=3, linestyle='-', 
                label=f"{result['naca']} Final Optimized (L/D={optimized_peak_ld_values[result_idx]:.1f})", alpha=0.9)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Top 3 Airfoil Shapes (Original + Stage 1 + Final)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Plot 8: L/D vs alpha comparison (top 3)
    plt.subplot(3, 4, 8)
    alpha_range = np.linspace(-5, 15, 21)
    
    for i, result in enumerate(sorted_results):
        orig_ld_values = []
        orig_x, orig_y = result['original_airfoil']['xy_coordinates']
        orig_coords = np.column_stack([orig_x, orig_y])
        
        for alpha in alpha_range:
            try:
                results = neuralfoil.get_aero_from_coordinates(
                    coordinates=orig_coords,
                    alpha=alpha,
                    Re=optimizer.reynolds_number,
                    model_size="xxxlarge"
                )
                cl = float(results['CL'].item() if hasattr(results['CL'], 'item') else results['CL'])
                cd = float(results['CD'].item() if hasattr(results['CD'], 'item') else results['CD'])
                ld = cl / cd if cd > 0 else 0
                orig_ld_values.append(ld)
            except:
                orig_ld_values.append(0)
        
        opt_ld_values = []
        opt_x, opt_y = result['optimized_airfoil']['xy_coordinates']
        opt_coords = np.column_stack([opt_x, opt_y])
        
        for alpha in alpha_range:
            try:
                results = neuralfoil.get_aero_from_coordinates(
                    coordinates=opt_coords,
                    alpha=alpha,
                    Re=optimizer.reynolds_number,
                    model_size="xxxlarge"
                )
                cl = float(results['CL'].item() if hasattr(results['CL'], 'item') else results['CL'])
                cd = float(results['CD'].item() if hasattr(results['CD'], 'item') else results['CD'])
                ld = cl / cd if cd > 0 else 0
                opt_ld_values.append(ld)
            except:
                opt_ld_values.append(0)
        
        plt.plot(alpha_range, orig_ld_values, color=colors_3[i], linewidth=1, linestyle='--', 
                label=f"{result['naca']} Original (Peak={max(orig_ld_values):.1f})", alpha=0.6)
        plt.plot(alpha_range, opt_ld_values, color=colors_3[i], linewidth=2, linestyle='-', 
                label=f"{result['naca']} Optimized (Peak={max(opt_ld_values):.1f})", alpha=0.9)
    
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('L/D Ratio')
    plt.title('L/D vs Alpha (Top 3)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Performance summary table
    plt.subplot(3, 4, 9)
    plt.axis('off')
    
    table_data = []
    headers = ['Shape', 'Stage 1', 'Stage 2', 'Final L/D', 'Area Δ%']
    
    for result in sorted_results:
        stats = result['optimizer'].get_early_stopping_stats() if 'optimizer' in result else None
        if stats and stats['stage1_complete']:
            stage1_ld = f"{stats['stage1_best_peak_ld']:.1f}"
            stage2_fitness = f"{stats['stage2_best_fitness']:.1f}"
            area_improvement = f"{area_improvements[naca_results.index(result)]:+.1f}%"
        else:
            stage1_ld = "N/A"
            stage2_fitness = "N/A"
            area_improvement = "N/A"
        
        table_data.append([
            result['naca'],
            stage1_ld,
            stage2_fitness,
            f"{result['fitness']:.1f}",
            area_improvement
        ])
    
    table = plt.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for j in range(len(headers)):
        table[(1, j)].set_facecolor('lightcoral')
    
    plt.title('Two-Stage Performance Summary (Top 3)', fontweight='bold')
    
    # Plot 10: AF512 format comparison (top 3)
    plt.subplot(3, 4, 10)
    x_points = np.linspace(0, 1, 512)
    
    for i, result in enumerate(sorted_results):
        af512_data = result['optimized_airfoil']['af512_data']
        plt.plot(x_points, af512_data[:, 0], color=colors_3[i], linewidth=2, 
                label=f"{result['naca']} Upper", alpha=0.7, linestyle='-')
        plt.plot(x_points, af512_data[:, 1], color=colors_3[i], linewidth=2, 
                label=f"{result['naca']} Lower", alpha=0.7, linestyle='--')
    
    plt.xlabel('X Position')
    plt.ylabel('Distance from Chord')
    plt.title('AF512 Format (Top 3)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 11: Two-stage optimization progress
    plt.subplot(3, 4, 11)
    if 'optimizer' in best_naca_result and best_naca_result['optimizer'].fitness_history:
        generations = [h['generation'] for h in best_naca_result['optimizer'].fitness_history]
        best_fitnesses = [h['best_fitness'] for h in best_naca_result['optimizer'].fitness_history]
        avg_fitnesses = [h['avg_fitness'] for h in best_naca_result['optimizer'].fitness_history]
        
        # Separate stage 1 and stage 2
        stage1_gens = [g for g, h in zip(generations, best_naca_result['optimizer'].fitness_history) 
                       if 'stage' not in h or h.get('stage') == 1]
        stage1_fitnesses = [f for f, h in zip(best_fitnesses, best_naca_result['optimizer'].fitness_history) 
                           if 'stage' not in h or h.get('stage') == 1]
        
        stage2_gens = [g for g, h in zip(generations, best_naca_result['optimizer'].fitness_history) 
                       if h.get('stage') == 2]
        stage2_fitnesses = [f for f, h in zip(best_fitnesses, best_naca_result['optimizer'].fitness_history) 
                           if h.get('stage') == 2]
        
        if stage1_gens:
            plt.plot(stage1_gens, stage1_fitnesses, 'b-', linewidth=2, label='Stage 1 (Peak L/D)', marker='o')
        if stage2_gens:
            plt.plot(stage2_gens, stage2_fitnesses, 'r-', linewidth=2, label='Stage 2 (Area + Peak)', marker='s')
        
        plt.axhline(y=best_naca_result['original_fitness'], color='g', linestyle=':', linewidth=2, label='Original L/D')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title('Two-Stage Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 12: Peak L/D preservation check
    plt.subplot(3, 4, 12)
    preserved_shapes = []
    not_preserved_shapes = []
    
    for result in naca_results:
        stats = result['optimizer'].get_early_stopping_stats() if 'optimizer' in result else None
        if stats and stats['stage1_complete']:
            if stats['peak_ld_preserved']:
                preserved_shapes.append(result['naca'])
            else:
                not_preserved_shapes.append(result['naca'])
    
    if preserved_shapes or not_preserved_shapes:
        labels = ['Preserved', 'Not Preserved']
        sizes = [len(preserved_shapes), len(not_preserved_shapes)]
        colors = ['lightgreen', 'lightcoral']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
        plt.title('Peak L/D Preservation (98% threshold)')
    else:
        plt.text(0.5, 0.5, 'No two-stage data\navailable', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Peak L/D Preservation')
    
    # Add a new plot for L/D falloff characteristics
    plt.figure(figsize=(16, 10))
    plt.suptitle('L/D Falloff Analysis - Maintaining Performance at Higher Angles', fontsize=16, fontweight='bold')
    
    # Plot 1: L/D vs Alpha for all optimized airfoils
    plt.subplot(2, 2, 1)
    alpha_range = np.linspace(-5, 15, 21)
    
    for i, result in enumerate(naca_results):
        if 'ld_values' in result['optimized_airfoil']['aerodynamic_data']:
            ld_values = result['optimized_airfoil']['aerodynamic_data']['ld_values']
            if len(ld_values) >= 21:
                color = 'red' if result['naca'] == best_naca_result['naca'] else 'blue'
                plt.plot(alpha_range, ld_values, color=color, alpha=0.7, linewidth=2, 
                        label=f"{result['naca']} (Peak: {max(ld_values):.1f})")
    
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='10° Check Point')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('L/D Ratio')
    plt.title('L/D vs Alpha - All Optimized Airfoils')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: L/D falloff ratio at 10 degrees
    plt.subplot(2, 2, 2)
    falloff_ratios = []
    shape_names_falloff = []
    
    for result in naca_results:
        if 'ld_values' in result['optimized_airfoil']['aerodynamic_data']:
            ld_values = result['optimized_airfoil']['aerodynamic_data']['ld_values']
            if len(ld_values) >= 21:
                peak_ld = max(ld_values)
                ld_at_10deg = ld_values[15]  # Index 15 = 10 degrees
                if peak_ld > 0:
                    falloff_ratio = ld_at_10deg / peak_ld
                    falloff_ratios.append(falloff_ratio)
                    shape_names_falloff.append(result['naca'])
    
    if falloff_ratios:
        colors_falloff = ['red' if name == best_naca_result['naca'] else 'blue' for name in shape_names_falloff]
        bars = plt.bar(range(len(shape_names_falloff)), falloff_ratios, color=colors_falloff, alpha=0.7)
        
        # Add 90% threshold line
        plt.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% Threshold')
        
        # Add value labels
        for bar, ratio in zip(bars, falloff_ratios):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{ratio:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Starting Shape')
        plt.ylabel('L/D Ratio at 10° / Peak L/D')
        plt.title('L/D Falloff Ratio at 10° (Target: ≥0.9)')
        plt.xticks(range(len(shape_names_falloff)), shape_names_falloff, rotation=45)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: L/D values at specific angles
    plt.subplot(2, 2, 3)
    angles_to_check = [0, 5, 10, 15]
    angle_indices = [5, 10, 15, 20]  # Corresponding indices in -5 to 15 range
    
    for i, result in enumerate(naca_results):
        if 'ld_values' in result['optimized_airfoil']['aerodynamic_data']:
            ld_values = result['optimized_airfoil']['aerodynamic_data']['ld_values']
            if len(ld_values) >= 21:
                ld_at_angles = []
                for idx in angle_indices:
                    if idx < len(ld_values):
                        ld_at_angles.append(ld_values[idx])
                    else:
                        ld_at_angles.append(0)
                
                color = 'red' if result['naca'] == best_naca_result['naca'] else 'blue'
                plt.plot(angles_to_check, ld_at_angles, color=color, alpha=0.7, linewidth=2, 
                        marker='o', label=f"{result['naca']}")
    
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('L/D Ratio')
    plt.title('L/D Performance at Key Angles')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Falloff penalty summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['Shape', 'Peak L/D', 'L/D at 10°', 'Falloff %', 'Status']
    
    for result in naca_results:
        if 'ld_values' in result['optimized_airfoil']['aerodynamic_data']:
            ld_values = result['optimized_airfoil']['aerodynamic_data']['ld_values']
            if len(ld_values) >= 21:
                peak_ld = max(ld_values)
                ld_at_10deg = ld_values[15]
                if peak_ld > 0:
                    falloff_ratio = ld_at_10deg / peak_ld
                    falloff_percent = (1 - falloff_ratio) * 100
                    status = "Good" if falloff_ratio >= 0.9 else "Poor"
                    
                    table_data.append([
                        result['naca'],
                        f"{peak_ld:.2f}",
                        f"{ld_at_10deg:.2f}",
                        f"{falloff_percent:.1f}%",
                        status
                    ])
    
    if table_data:
        table = plt.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code the status column
        for i, row in enumerate(table_data):
            if row[4] == "Good":
                table[(i+1, 4)].set_facecolor('lightgreen')
            else:
                table[(i+1, 4)].set_facecolor('lightcoral')
    
    plt.title('L/D Falloff Summary', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"L/D falloff analysis displayed")
    
    plt.tight_layout()
    plt.show()
    
    print(f"Comprehensive two-stage comparison displayed")

def main():
    print("Welcome to the NACA Airfoil Shape Optimizer!")
    print("Start with a NACA 4-digit profile (e.g., 2412) or ellipse-based shape!")
    print("Four-Stage Optimization:")
    print("  Stage 1: Maximize peak L/D ratio")
    print("  Stage 2: Maximize peak CL at peak L/D (with 2% max peak L/D drop)")
    print("  Stage 3: Maximize average CL over ±4° range (with 2% max drops)")
    print("  Stage 4: Maximize average L/D over ±4° range (with 2% max drops)")
    print()
    
    while True:
        try:
            original, optimized = optimize_naca_and_compare()
            
            print(f"\n" + "="*50)
            choice = input("Optimize another NACA? (y/n): ").strip().lower()
            if choice not in ['y', 'yes']:
                break
            print()
            
        except KeyboardInterrupt:
            print(f"\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")
            continue
    
    print(f"Thanks for using the NACA Optimizer!")

if __name__ == "__main__":
    main()

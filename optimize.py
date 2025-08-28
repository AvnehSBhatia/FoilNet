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

WINGSPAN = 10.0
CHORD = 1.0
AIRSPEED = 100.0

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
        pop_size, gens = 30, 15
        early_stop_patience, early_stop_tolerance = 5, 0.001
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
    
    print(f"\nStage 1: Optimizing ellipse-based starting shapes...")
    
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
    
    starting_shapes = [
        ('Ellipse (12% thickness, centered)', create_ellipse_af512(0.12, 0.5)),
    ]
    
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
            naca_code = shape_name.split()[-1]
            original_individual = optimizer.create_individual_from_naca(naca_code)
        
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
            optimized_airfoil = temp_optimizer.run_optimization(naca_code)
        
        naca_results.append({
            'naca': shape_name,
            'original_airfoil': original_individual,
            'original_fitness': original_fitness,
            'optimized_airfoil': optimized_airfoil,
            'fitness': optimized_airfoil['fitness'],
            'optimizer': temp_optimizer
        })
        
        print(f"   {shape_name}: Original L/D = {original_fitness:.3f} → Optimized L/D = {optimized_airfoil['fitness']:.3f}")
        print(f"   Improvement: {((optimized_airfoil['fitness'] - original_fitness) / original_fitness * 100):.1f}%")
    
    best_naca_result = max(naca_results, key=lambda x: x['fitness'])
    best_shape_name = best_naca_result['naca']
    best_optimized_airfoil = best_naca_result['optimized_airfoil']
    
    print(f"\nBest optimized shape: {best_shape_name} with L/D = {best_naca_result['fitness']:.3f}")
    
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
                print(f"  - Best L/D: {stats['best_fitness']:.3f} at generation {stats['best_fitness_generation']}")
                
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
    print(f"OPTIMIZATION RESULTS")
    print(f"="*50)
    print(f"Best optimized shape: {best_shape_name} with L/D = {best_naca_result['fitness']:.3f}")
    
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
    
    fig.suptitle(f'Flight Conditions: {optimizer.wingspan}ft wingspan, {optimizer.chord}ft chord, {optimizer.airspeed}mph airspeed (Re ≈ {optimizer.reynolds_number:.0f}) | Thickness: 8-16%', 
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
    plt.ylabel('L/D Ratio')
    plt.title('Performance Comparison')
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
        
        plt.plot(generations, best_fitnesses, 'b-', linewidth=2, label='Best L/D', marker='o')
        plt.plot(generations, avg_fitnesses, 'r--', linewidth=2, label='Average L/D', alpha=0.7)
        plt.axhline(y=original['fitness'], color='g', linestyle=':', linewidth=2, label='Original L/D')
        plt.xlabel('Generation')
        plt.ylabel('L/D Ratio')
        plt.title('Optimization Progress')
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
    
    clean_name = best_shape_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    filename_base = f"best_optimized_{clean_name}_L{fitness:.1f}"
    
    dat_filename = f"{filename_base}.dat"
    with open(dat_filename, 'w') as f:
        f.write(f"Optimized {best_shape_name} Airfoil\n")
        f.write(f"L/D = {fitness:.3f}, Peak Alpha = {aero_data['peak_alpha']:.2f}°\n")
        f.write(f"CL = {aero_data['CL']:.4f}, CD = {aero_data['CD']:.4f}\n")
        f.write(f"Flight: {optimizer.wingspan}ft wingspan, {optimizer.chord}ft chord, {optimizer.airspeed}mph\n")
        f.write(f"Reynolds Number: {optimizer.reynolds_number:.0f}\n")
        f.write(f"{len(x_coords)}\n")
        
        for i in range(len(x_coords)):
            f.write(f"{x_coords[i]:.6f} {y_coords[i]:.6f}\n")
    
    print(f"   DAT file saved as: {dat_filename}")
    
    dxf_filename = f"{filename_base}.dxf"
    convert_dat_to_dxf(dat_filename, dxf_filename, best_shape_name, fitness)
    
    print(f"DAT and DXF files saved successfully!")

def create_comprehensive_plots(naca_results, best_naca_result, optimizer, target_lift, early_stop_patience=None, early_stop_tolerance=None):
    print(f"\nCreating comprehensive comparison plots...")
    
    shape_names = [result['naca'] for result in naca_results]
    original_fitness_values = [result['original_fitness'] for result in naca_results]
    optimized_fitness_values = [result['fitness'] for result in naca_results]
    peak_alphas = [result['optimized_airfoil']['aerodynamic_data']['peak_alpha'] for result in naca_results]
    cl_values = [result['optimized_airfoil']['aerodynamic_data']['CL'] for result in naca_results]
    cd_values = [result['optimized_airfoil']['aerodynamic_data']['CD'] for result in naca_results]
    
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
    
    fig = plt.figure(figsize=(20, 12))
    
    title = f'Multi-NACA Optimization Results | Flight: {optimizer.wingspan}ft wingspan, {optimizer.chord}ft chord, {optimizer.airspeed}mph'
    if target_lift:
        title += f' | Target Lift: {target_lift:.3f} CL'
    if early_stop_patience and early_stop_tolerance:
        title += f' | Early Stop: {early_stop_patience} gens, tol={early_stop_tolerance}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.subplot(2, 4, 1)
    x = np.arange(len(shape_names))
    width = 0.35
    
    original_bars = plt.bar(x - width/2, original_peak_ld_values, width, label='Original', alpha=0.7, color='lightblue')
    optimized_bars = plt.bar(x + width/2, optimized_peak_ld_values, width, label='Optimized', alpha=0.7, color='orange')
    
    best_idx = shape_names.index(best_naca_result['naca'])
    optimized_bars[best_idx].set_color('red')
    
    plt.xlabel('Starting Shape')
    plt.ylabel('L/D Ratio')
    plt.title('L/D Performance: Original vs Optimized')
    plt.xticks(x, shape_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, (orig_bar, opt_bar) in enumerate(zip(original_bars, optimized_bars)):
        plt.text(orig_bar.get_x() + orig_bar.get_width()/2, orig_bar.get_height() + 0.5, 
                f'{original_peak_ld_values[i]:.1f}', ha='center', va='bottom', fontsize=8)
        plt.text(opt_bar.get_x() + opt_bar.get_width()/2, opt_bar.get_height() + 0.5, 
                f'{optimized_peak_ld_values[i]:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.subplot(2, 4, 2)
    colors_alpha = ['red' if result['naca'] == best_naca_result['naca'] else 'blue' for result in naca_results]
    plt.bar(range(len(shape_names)), peak_alphas, color=colors_alpha, alpha=0.7)
    plt.xlabel('Starting Shape')
    plt.ylabel('Peak Alpha (degrees)')
    plt.title('Optimal Angle of Attack')
    plt.xticks(range(len(shape_names)), shape_names, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(2, 4, 3)
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
    
    plt.subplot(2, 4, 4)
    
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
    
    plt.subplot(2, 4, 5)
    sorted_results = sorted(naca_results, key=lambda x: x['fitness'], reverse=True)[:3]
    colors_3 = ['red', 'orange', 'green']
    
    for i, result in enumerate(sorted_results):
        result_idx = naca_results.index(result)
        
        orig_x, orig_y = result['original_airfoil']['xy_coordinates']
        plt.plot(orig_x, orig_y, color=colors_3[i], linewidth=1, linestyle='--', 
                label=f"{result['naca']} Original (L/D={original_peak_ld_values[result_idx]:.1f})", alpha=0.6)
        
        opt_x, opt_y = result['optimized_airfoil']['xy_coordinates']
        plt.plot(opt_x, opt_y, color=colors_3[i], linewidth=3, linestyle='-', 
                label=f"{result['naca']} Optimized (L/D={optimized_peak_ld_values[result_idx]:.1f})", alpha=0.9)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Top 3 Airfoil Shapes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.subplot(2, 4, 6)
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
    
    plt.subplot(2, 4, 7)
    plt.axis('off')
    
    table_data = []
    headers = ['Shape', 'Original L/D', 'Optimized L/D', 'Improvement', 'Peak α', 'CL']
    
    for result in sorted_results:
        improvement = ((result['fitness'] - result['original_fitness']) / result['original_fitness']) * 100
        table_data.append([
            result['naca'],
            f"{result['original_fitness']:.1f}",
            f"{result['fitness']:.1f}",
            f"+{improvement:.1f}%",
            f"{result['optimized_airfoil']['aerodynamic_data']['peak_alpha']:.1f}°",
            f"{result['optimized_airfoil']['aerodynamic_data']['CL']:.3f}"
        ])
    
    table = plt.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for j in range(len(headers)):
        table[(1, j)].set_facecolor('lightcoral')
    
    plt.title('Performance Summary (Top 3)', fontweight='bold')
    
    plt.subplot(2, 4, 8)
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
    
    plt.tight_layout()
    plt.show()
    
    print(f"Comprehensive comparison displayed")

def main():
    print("Welcome to the Airfoil Shape Optimizer!")
    print("Starting with ellipse-based shapes, watch them get optimized for maximum L/D ratio!")
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
            print(f"\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")
            continue
    
    print(f"Thanks for using the NACA Optimizer!")

if __name__ == "__main__":
    main()

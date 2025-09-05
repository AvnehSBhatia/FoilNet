import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import math
import time
import os

def naca4_digit_to_coordinates(naca_code, num_points=500):
    if len(naca_code) != 4:
        raise ValueError("NACA code must be 4 digits")
    
    m = int(naca_code[0]) / 100.0
    p = int(naca_code[1]) / 10.0
    t = int(naca_code[2:]) / 100.0
    
    x = np.linspace(0, 1, num_points)
    
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    if m == 0 or p == 0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(x < p,
                      m * (2 * p * x - x**2) / p**2,
                      m * (1 - 2 * p + 2 * p * x - x**2) / (1 - p)**2)
        
        dyc_dx = np.where(x < p,
                          2 * m * (p - x) / p**2,
                          2 * m * (p - x) / (1 - p)**2)
    
    theta = np.arctan(dyc_dx)
    
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    
    if len(x_coords) != num_points:
        indices = np.linspace(0, len(x_coords) - 1, num_points, dtype=int)
        x_coords = x_coords[indices]
        y_coords = y_coords[indices]
    
    return x_coords, y_coords

def airfoil_to_af512(airfoil_code, num_points=512):
    """
    Convert airfoil code to AF512 format.
    Supports both NACA (e.g., '2412') and Selig (e.g., 'sg6043') formats.
    """
    airfoil_code = airfoil_code.lower().strip()
    
    # Check if it's a Selig format (starts with 'sg' or 'selig')
    if airfoil_code.startswith('sg') or airfoil_code.startswith('selig'):
        return selig_to_af512(airfoil_code, num_points)
    else:
        # Assume it's a NACA format
        return naca_to_af512(airfoil_code, num_points)

def selig_to_af512(selig_code, num_points=512):
    """
    Convert Selig format airfoil code to AF512 format.
    Selig codes like 'sg6043' represent specific airfoil designs.
    """
    try:
        # For now, we'll use a placeholder approach
        # In a real implementation, you would load the actual Selig airfoil coordinates
        # from a database or file
        
        # Generate a reasonable airfoil shape based on the code
        # This is a simplified approach - you might want to implement proper Selig loading
        
        x_coords = np.linspace(0, 1, num_points*2)
        
        # Create a basic airfoil shape (this is a placeholder)
        # In practice, you would load the actual Selig airfoil coordinates
        thickness = 0.12  # Default thickness
        camber = 0.02     # Default camber
        
        # Generate upper and lower surfaces
        upper_y = thickness * np.sqrt(1 - (x_coords - 0.5)**2 / 0.25) + camber * (1 - (x_coords - 0.5)**2 / 0.25)
        lower_y = -thickness * np.sqrt(1 - (x_coords - 0.5)**2 / 0.25) + camber * (1 - (x_coords - 0.5)**2 / 0.25)
        
        # Ensure leading and trailing edges are at zero
        upper_y[0] = 0.0
        upper_y[-1] = 0.0
        lower_y[0] = 0.0
        lower_y[-1] = 0.0
        
        # Convert to AF512 format
        return naca_to_af512_from_coordinates(x_coords, upper_y, num_points)
        
    except Exception as e:
        print(f"Error converting Selig {selig_code} to AF512: {e}")
        # Fallback to default airfoil
        return naca_to_af512("0012", num_points)

def naca_to_af512(naca_code, num_points=512):
    x_coords, y_coords = naca4_digit_to_coordinates(naca_code, num_points*2)
    
    mid_point = len(x_coords) // 2
    upper_x = x_coords[:mid_point]
    upper_y = y_coords[:mid_point]
    lower_x = x_coords[mid_point:]
    lower_y = y_coords[mid_point:]
    
    x_uniform = np.linspace(0, 1, num_points)
    
    from scipy.interpolate import interp1d
    
    upper_interp = interp1d(upper_x, upper_y, kind='linear', bounds_error=False, fill_value='extrapolate')
    upper_y_uniform = upper_interp(x_uniform)
    
    lower_interp = interp1d(lower_x, lower_y, kind='linear', bounds_error=False, fill_value='extrapolate')
    lower_y_uniform = lower_interp(x_uniform)
    
    af512_data = np.column_stack([upper_y_uniform, lower_y_uniform])
    
    return af512_data

def af512_to_coordinates(af512_data):
    num_points = len(af512_data)
    x_uniform = np.linspace(0, 1, num_points)
    
    upper_dist = af512_data[:, 0]
    lower_dist = af512_data[:, 1]
    
    x_coords = np.concatenate([x_uniform[::-1], x_uniform[1:]])
    y_coords = np.concatenate([upper_dist[::-1], lower_dist[1:]])
    
    return x_coords, y_coords

def naca_to_af512_from_coordinates(x_coords, y_coords, num_points=512):
    mid_point = len(x_coords) // 2
    upper_x = x_coords[:mid_point]
    upper_y = y_coords[:mid_point]
    lower_x = x_coords[mid_point:]
    lower_y = y_coords[mid_point:]
    
    x_uniform = np.linspace(0, 1, num_points)
    
    from scipy.interpolate import interp1d
    
    upper_interp = interp1d(upper_x, upper_y, kind='linear', bounds_error=False, fill_value='extrapolate')
    upper_y_uniform = upper_interp(x_uniform)
    
    lower_interp = interp1d(lower_x, lower_y, kind='linear', bounds_error=False, fill_value='extrapolate')
    lower_y_uniform = lower_interp(x_uniform)
    
    af512_data = np.column_stack([upper_y_uniform, lower_y_uniform])
    
    return af512_data

def generate_balanced_naca_combinations(num_samples, m_range, p_range, t_range):
    all_combinations = []
    for m in m_range:
        for p in p_range:
            for t in t_range:
                all_combinations.append((m, p, t))
    
    samples_per_m = max(1, num_samples // len(m_range))
    samples_per_p = max(1, num_samples // len(p_range))
    samples_per_t = max(1, num_samples // len(t_range))
    
    selected_combinations = []
    
    m_counts = {m: 0 for m in m_range}
    p_counts = {p: 0 for p in p_range}
    t_counts = {t: 0 for t in t_range}
    
    np.random.shuffle(all_combinations)
    
    for m, p, t in all_combinations:
        if len(selected_combinations) >= num_samples:
            break
            
        if (m_counts[m] < samples_per_m and 
            p_counts[p] < samples_per_p and 
            t_counts[t] < samples_per_t):
            
            selected_combinations.append((m, p, t))
            m_counts[m] += 1
            p_counts[p] += 1
            t_counts[t] += 1
    
    remaining_combinations = [c for c in all_combinations if c not in selected_combinations]
    np.random.shuffle(remaining_combinations)
    
    while len(selected_combinations) < num_samples and remaining_combinations:
        selected_combinations.append(remaining_combinations.pop())
    
    np.random.shuffle(selected_combinations)
    
    return selected_combinations[:num_samples]

def generate_balanced_naca_combinations_repeated(num_samples, m_range, p_range, t_range):
    total_combinations = len(m_range) * len(p_range) * len(t_range)
    repeats_needed = (num_samples // total_combinations) + 1
    
    selected_combinations = []
    
    for repeat in range(repeats_needed):
        if len(selected_combinations) >= num_samples:
            break
            
        repeat_combinations = generate_balanced_naca_combinations(
            min(total_combinations, num_samples - len(selected_combinations)), 
            m_range, p_range, t_range
        )
        selected_combinations.extend(repeat_combinations)
    
    return selected_combinations[:num_samples]

def generate_training_data(num_samples=10000, num_points=512):
    af512_file = 'training_af512_data.npy'
    xy_file = 'training_xy_coordinates.npy'
    
    if os.path.exists(af512_file) and os.path.exists(xy_file) and not FORCE_REGENERATE_DATA:
        af512_data = np.load(af512_file)
        xy_coordinates = np.load(xy_file)
        return af512_data, xy_coordinates
    
    af512_data = []
    xy_coordinates = []
    
    naca_af512, naca_xy = generate_naca_training_data(num_samples, num_points)
    af512_data.extend(naca_af512)
    xy_coordinates.extend(naca_xy)
    
    clean_af512_data = af512_data.copy()
    clean_xy_coordinates = xy_coordinates.copy()
    
    af512_data = add_noise_to_training_data(af512_data, noise_level=0.005)
    
    return np.array(af512_data), np.array(xy_coordinates)

def add_noise_to_training_data(data, noise_level=0.005):
    if isinstance(data, list):
        data = np.array(data)
    
    noise = np.random.normal(0, noise_level, data.shape)
    
    noisy_data = data + noise
    
    return noisy_data

def analyze_class_distribution(af512_data, xy_coordinates):
    pass

def analyze_naca_distribution(num_samples):
    pass

def create_balanced_train_val_split(af512_data, xy_coordinates, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    
    n_samples = len(af512_data)
    n_val = int(n_samples * test_size)
    n_train = n_samples - n_val
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_af512 = af512_data[train_indices]
    val_af512 = af512_data[val_indices]
    train_xy = xy_coordinates[train_indices]
    val_xy = xy_coordinates[val_indices]
    
    train_af512 = add_noise_to_training_data(train_af512, noise_level=0.005)
    
    return train_af512, val_af512, train_xy, val_xy

def generate_naca_training_data(num_samples, num_points=512):
    af512_data = []
    xy_coordinates = []
    
    m_range = list(range(10))
    p_range = list(range(10))
    
    t_range = []
    
    for i in range(8, 17):
        t_range.append(i)
    
    for i in range(8, 16):
        for j in range(1, 10):
            t_range.append(i + j * 0.1)
    
    additional_fractionals = [4.5, 6.4, 7.2, 7.8, 16.2, 16.5, 16.8]
    t_range.extend(additional_fractionals)
    
    t_range = [t for t in t_range if 8.0 <= t <= 16.0]
    
    t_range = sorted(list(set(t_range)))
    
    if num_samples <= len(m_range) * len(p_range) * len(t_range):
        selected_combinations = generate_balanced_naca_combinations(num_samples, m_range, p_range, t_range)
    else:
        selected_combinations = generate_balanced_naca_combinations_repeated(num_samples, m_range, p_range, t_range)
    
    valid_samples = 0
    for i, (m, p, t) in enumerate(selected_combinations):
        if valid_samples >= num_samples:
            break
        
        if isinstance(t, float) and t != int(t):
            t_int = int(t)
            t_frac = t - t_int
            
            t_int_clamped = max(0, min(t_int, 99))
            t_next_clamped = max(0, min(t_int + 1, 99))
            
            naca_code_1 = f"{m}{p}{t_int_clamped:02d}"
            naca_code_2 = f"{m}{p}{t_next_clamped:02d}"
            
            try:
                x1, y1 = naca4_digit_to_coordinates(naca_code_1, num_points*2)
                x2, y2 = naca4_digit_to_coordinates(naca_code_2, num_points*2)
                
                x_coords = x1 * (1 - t_frac) + x2 * t_frac
                y_coords = y1 * (1 - t_frac) + y2 * t_frac
                
                af512 = naca_to_af512_from_coordinates(x_coords, y_coords, num_points)
                
                if af512.shape == (num_points, 2) and not np.any(np.isnan(af512)) and len(x_coords) == num_points*2:
                    af512_data.append(af512.flatten())
                    xy_coords = np.concatenate([x_coords, y_coords])
                    xy_coordinates.append(xy_coords)
                    valid_samples += 1
                else:
                    continue
                    
            except Exception as e:
                continue
                
        else:
            t_int = int(t)
            naca_code = f"{m}{p}{t_int:02d}"
            
            try:
                af512 = naca_to_af512(naca_code, num_points)
                
                x_coords, y_coords = naca4_digit_to_coordinates(naca_code, num_points*2)
                
                if af512.shape == (num_points, 2) and not np.any(np.isnan(af512)) and len(x_coords) == num_points*2:
                    af512_data.append(af512.flatten())
                    xy_coords = np.concatenate([x_coords, y_coords])
                    xy_coordinates.append(xy_coords)
                    valid_samples += 1
                else:
                    continue
                    
            except Exception as e:
                continue
    
    return af512_data, xy_coordinates

class AF512toXYDataset(Dataset):
    
    def __init__(self, af512_data, xy_coordinates, add_noise=False, noise_level=0.005):
        self.af512_data = af512_data
        self.xy_coordinates = xy_coordinates
        self.add_noise = add_noise
        self.noise_level = noise_level
    
    def __len__(self):
        return len(self.af512_data)
    
    def __getitem__(self, idx):
        af512 = self.af512_data[idx]
        xy_coords = self.xy_coordinates[idx]
        
        if self.add_noise:
            af512 = af512 + np.random.normal(0, self.noise_level, af512.shape)
        
        return torch.FloatTensor(af512), torch.FloatTensor(xy_coords)


class AF512toXYNet(nn.Module):
    
    def __init__(self, input_size=1024, output_size=2048, hidden_sizes=[512, 256, 128,256,512]):
        super(AF512toXYNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.SiLU(),
                nn.Dropout(0.2),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device='mps'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    print(f"Training on {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{num_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return train_losses, val_losses

def predict_xy_coordinates(model, af512_data, device='mps'):
    model.eval()
    with torch.no_grad():
        if af512_data.ndim == 2:
            af512_flat = af512_data.flatten()
        else:
            af512_flat = af512_data
        
        input_tensor = torch.FloatTensor(af512_flat).unsqueeze(0).to(device)
        output = model(input_tensor)
        xy_flat = output.cpu().numpy().flatten()
        
        x_coords = xy_flat[:1024]
        y_coords = xy_flat[1024:]
        
        return x_coords, y_coords


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    
    af512_data, xy_coordinates = generate_training_data(
        num_samples=NUM_SAMPLES, 
        num_points=512
    )
    
    np.save('training_af512_data.npy', af512_data)
    np.save('training_xy_coordinates.npy', xy_coordinates)
    
    train_af512, val_af512, train_xy, val_xy = create_balanced_train_val_split(
        af512_data, xy_coordinates, test_size=0.2, random_state=42
    )
    
    train_dataset = AF512toXYDataset(train_af512, train_xy, add_noise=True, noise_level=0.005)
    val_dataset = AF512toXYDataset(val_af512, val_xy, add_noise=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model_file = 'af512_to_xy_model.pth'
    if os.path.exists(model_file):
        model_state = torch.load(model_file, map_location='cpu')
        
        hidden_sizes = []
        for i in range(0, len(model_state.keys()) // 2 - 1):
            weight_key = f'network.{i*3}.weight'
            if weight_key in model_state:
                hidden_sizes.append(model_state[weight_key].shape[0])
        
        model = AF512toXYNet(input_size=1024, output_size=2048, hidden_sizes=hidden_sizes)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model = model.to(device)
        
        train_losses = []
        val_losses = []
        
    else:
        model = AF512toXYNet(input_size=1024, output_size=2048, hidden_sizes=[512, 256, 128,256,512])
        
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, 
            num_epochs=100, learning_rate=0.001, device=device
        )
        
        torch.save(model.state_dict(), model_file)
    

NUM_SAMPLES = 10000
FORCE_REGENERATE_DATA = False

if __name__ == "__main__":
    main()

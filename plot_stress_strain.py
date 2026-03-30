import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ==========================================
# 1. USER VARIABLES (Check these paths!)
# ==========================================
machine_csv_path = "BL_1.CSV"

# Make sure this points to your actual DIC output CSV file!
# It might be in "DIC_Output/strain_results.csv" or "Bl_1/strain_results.csv"
dic_csv_path = "DIC_Output/strain_results.csv" 

video_fps = 30.0  # Camera frame rate
width_mm = 10.0   # Specimen width
thickness_mm = 2.0 # Specimen thickness

# ==========================================
# 2. LOAD & CALCULATE MACHINE STRESS
# ==========================================
print("Loading Machine Data...")
machine_df = pd.read_csv(machine_csv_path, sep=None, engine='python')
machine_df.columns = machine_df.columns.str.strip()

machine_time = machine_df['Time (Sec.)'].values
machine_load_kn = machine_df['Load (kN)'].values

# Convert Load to Newtons and calculate Stress (MPa)
force_n = machine_load_kn * 1000.0
stress_mpa = force_n / (width_mm * thickness_mm)

# ==========================================
# 3. LOAD DIC STRAIN & SYNCHRONIZE
# ==========================================
print("Loading DIC Strain Data...")
dic_df = pd.read_csv(dic_csv_path)
dic_strain_eyy = dic_df['eyy'].values

# Create timestamps for the DIC frames based on camera FPS
dic_time = np.arange(len(dic_df)) / video_fps

# Interpolate: Align the machine stress precisely to the camera's timestamps
print("Synchronizing Data...")
stress_interpolator = interp1d(
    machine_time, 
    stress_mpa, 
    kind='linear', 
    bounds_error=False, 
    fill_value=np.nan
)
synced_stress = stress_interpolator(dic_time)

# ==========================================
# 4. PLOT THE CURVE
# ==========================================
plt.figure(figsize=(10, 6))

# Plot Strain on X, Stress on Y
plt.plot(dic_strain_eyy, synced_stress, 'b-', linewidth=2)

plt.title("Engineering Stress-Strain Curve")
plt.xlabel("Strain (eyy)")
plt.ylabel("Stress (MPa)")
plt.grid(True)

plt.tight_layout()
plt.show()
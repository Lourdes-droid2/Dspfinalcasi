import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys  # Necesario para usar sys.exit()

sns.set(style="whitegrid")

# --- Cargar datos desde el archivo correcto ---
csv_filename = "doa_array_avg_results.csv"
print(f"Attempting to load {csv_filename}...")
try:
    df = pd.read_csv(csv_filename)
    print("CSV loaded successfully.")
    print("Shape of df:", df.shape)
    print("Head of df:\n", df.head())
    print("Columns in df:", df.columns.tolist())
    print("Data types of df columns:\n", df.dtypes)
except FileNotFoundError:
    print(f"ERROR: {csv_filename} not found! Please run main.py to generate it.")
    sys.exit()
except Exception as e:
    print(f"ERROR loading {csv_filename}: {e}")
    sys.exit()

# --- Filtrar datos válidos ---
print("\nFiltering df_array...")
df_array = df[df['doa_array_error_deg'].notna()].copy()
print("Shape of df_array:", df_array.shape)
print("Head of df_array:\n", df_array.head())

# --- Verificar columnas necesarias ---
required_cols = [
    'actual_elevation_src_to_array_center_deg',
    'actual_dist_src_to_array_center_m',
    'mic_separation_m',
    'num_mics_processed',
    'rt60_target_s',
    'fs_hz'
]

if 'actual_elevation_src_to_array_center_deg' not in df_array.columns and 'elevation_angle_deg' in df_array.columns:
    print("Info: 'actual_elevation_src_to_array_center_deg' not found, renaming from 'elevation_angle_deg'.")
    df_array.rename(columns={'elevation_angle_deg': 'actual_elevation_src_to_array_center_deg'}, inplace=True)

missing_cols = [col for col in required_cols if col not in df_array.columns]
if missing_cols:
    print(f"WARNING: Missing columns in df_array: {missing_cols}")
    print("Available columns:", df_array.columns.tolist())

# --- Convertir columnas a numéricas ---
print("\nConverting columns to numeric...")
for col in required_cols:
    if col in df_array.columns:
        df_array[col] = pd.to_numeric(df_array[col], errors='coerce')
        print(f"Converted {col}. NaN count: {df_array[col].isna().sum()}")

# --- Función general para graficar ---
def plot_metric_vs_param(df_plot, param_col, title, xlabel):
    print(f"\nGenerating plot: {title}")
    if df_plot.empty or param_col not in df_plot.columns or df_plot[param_col].isna().all():
        print(f"Skipping plot '{title}' — column missing or empty.")
        return

    plt.figure(figsize=(8, 5))
    if 'tdoa_method_for_avg_doa' in df_plot.columns and df_plot['tdoa_method_for_avg_doa'].notna().any():
        sns.lineplot(
            data=df_plot,
            x=param_col,
            y='doa_array_error_deg',
            hue='tdoa_method_for_avg_doa',
            errorbar='sd',
            marker='o'
        )
        plt.legend(title="Método TDOA")
    else:
        sns.lineplot(
            data=df_plot,
            x=param_col,
            y='doa_array_error_deg',
            errorbar='sd',
            marker='o'
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Error promedio DOA (grados)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Gráficos por variable ---
plot_metric_vs_param(df_array, 'actual_elevation_src_to_array_center_deg',
                     "Error promedio DOA vs Ángulo de elevación", "Ángulo de elevación (grados)")

plot_metric_vs_param(df_array, 'actual_dist_src_to_array_center_m',
                     "Error promedio DOA vs Distancia fuente-arreglo", "Distancia (m)")

plot_metric_vs_param(df_array, 'mic_separation_m',
                     "Error promedio DOA vs Separación entre micrófonos", "Separación entre micrófonos (m)")

plot_metric_vs_param(df_array, 'num_mics_processed',
                     "Error promedio DOA vs Cantidad de micrófonos", "Cantidad de micrófonos")

plot_metric_vs_param(df_array, 'rt60_target_s',
                     "Error promedio DOA vs Tiempo de reverberación RT60", "Tiempo RT60 (s)")

plot_metric_vs_param(df_array, 'fs_hz',
                     "Error promedio DOA vs Frecuencia de muestreo", "Frecuencia de muestreo (Hz)")

plot_metric_vs_param(df_array, 'actual_azimuth_src_to_array_center_deg',
                     "Error promedio DOA vs Ángulo azimuth", "Ángulo azimuth (grados)")

# --- Último gráfico: Error promedio vs SNR por frecuencia de muestreo ---
title_last = "Error promedio DOA vs SNR para distintas frecuencias de muestreo"
print(f"\nGenerating plot: {title_last}")
if 'snr_db' in df_array.columns and 'fs_hz' in df_array.columns:
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_array,
        x='snr_db',
        y='doa_array_error_deg',
        hue='fs_hz',
        errorbar='sd',
        marker='o',
        palette='viridis'
    )
    plt.title(title_last)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Error promedio DOA (grados)")
    plt.legend(title="Frecuencia de muestreo (Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print(f"Skipping plot '{title_last}' — 'snr_db' or 'fs_hz' missing.")

print("\n--- Script generate_plots.py finished ---")

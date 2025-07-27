import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sns.set(style="whitegrid")
plt.style.use('default')

# Cargar CSV
try:
    df = pd.read_csv("doa_array_avg_results.csv")
except Exception as e:
    print("❌ Error al cargar el archivo CSV:", e)
    sys.exit()

# Filtrar datos válidos
df_array = df[(df['mic_pair'] == 'array_avg_adj_pairs') & df['doa_array_error_deg'].notna()].copy()

# Renombrar si falta elevación
if 'actual_elevation_src_to_array_center_deg' not in df_array.columns and 'elevation_deg' in df_array.columns:
    df_array.rename(columns={'elevation_deg': 'actual_elevation_src_to_array_center_deg'}, inplace=True)

# Asegurar tipos numéricos
numeric_cols = [
    'actual_elevation_src_to_array_center_deg',
    'actual_dist_src_to_array_center_m',
    'mic_separation_m',
    'num_mics_processed',
    'rt60_target_s',
    'fs_hz',
    'actual_azimuth_src_to_array_center_deg',
    'snr_db'
]
for col in numeric_cols:
    if col in df_array.columns:
        df_array[col] = pd.to_numeric(df_array[col], errors='coerce')

def clean_df_for_param(df, param_col):
    cols_needed = ['doa_array_error_deg', 'tdoa_method_for_avg_doa', param_col]
    return df.dropna(subset=cols_needed).copy()

# Función: gráfico promedio (lineplot)
def plot_lineplot(df, param_col, title, xlabel):
    if param_col not in df.columns or df[param_col].nunique() < 3:
        return
    if df['doa_array_error_deg'].nunique() <= 1:
        return

    df_clean = clean_df_for_param(df, param_col)

    df_avg = (
        df_clean.groupby([param_col, 'tdoa_method_for_avg_doa'], observed=True)['doa_array_error_deg']
        .mean()
        .reset_index()
    )

    methods_order = ['cc', 'phat', 'scot', 'roth', 'ml']
    df_avg['tdoa_method_for_avg_doa'] = pd.Categorical(df_avg['tdoa_method_for_avg_doa'], categories=methods_order, ordered=True)

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_avg,
        x=param_col,
        y='doa_array_error_deg',
        hue='tdoa_method_for_avg_doa',
        style='tdoa_method_for_avg_doa',
        markers=True,
        dashes=False,
        hue_order=methods_order,
        errorbar=None,
    )
    plt.title(title + " (Promedio)")
    plt.xlabel(xlabel)
    plt.ylabel("Error DOA (°)")
    plt.legend(title="Método TDOA", loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Función: gráfico boxplot
def plot_boxplot(df, param_col, title, xlabel):
    if param_col not in df.columns or df[param_col].nunique() < 2:
        return
    df_clean = clean_df_for_param(df, param_col)
    df_clean[param_col] = df_clean[param_col].astype(str)

    methods_order = ['cc', 'phat', 'scot', 'roth', 'ml']
    df_clean['tdoa_method_for_avg_doa'] = pd.Categorical(df_clean['tdoa_method_for_avg_doa'], categories=methods_order, ordered=True)

    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df_clean,
        x=param_col,
        y='doa_array_error_deg',
        hue='tdoa_method_for_avg_doa',
        hue_order=methods_order,
    )
    plt.title(title + " (Boxplot)")
    plt.xlabel(xlabel)
    plt.ylabel("Error DOA (°)")
    plt.legend(title="Método TDOA", loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Lista de parámetros
plots = [
    ('actual_elevation_src_to_array_center_deg', "Error DOA vs Ángulo de elevación", "Elevación (°)"),
    ('actual_dist_src_to_array_center_m', "Error DOA vs Distancia fuente-arreglo", "Distancia (m)"),
    ('mic_separation_m', "Error DOA vs Separación entre micrófonos", "Separación (m)"),
    ('num_mics_processed', "Error DOA vs Cantidad de micrófonos", "Cantidad de micrófonos"),
    ('rt60_target_s', "Error DOA vs RT60", "RT60 (s)"),
    ('fs_hz', "Error DOA vs Frecuencia de muestreo", "Frecuencia (Hz)"),
    ('actual_azimuth_src_to_array_center_deg', "Error DOA vs Ángulo azimuth", "Azimuth (°)"),
]

# Generar ambos tipos de gráficos
for param, title, xlabel in plots:
    plot_lineplot(df_array, param, title, xlabel)
    plot_boxplot(df_array, param, title, xlabel)

# SNR vs DOA (especial por frecuencia de muestreo)
if 'snr_db' in df_array.columns and 'fs_hz' in df_array.columns:
    if df_array['snr_db'].nunique() > 2:
        df_clean = clean_df_for_param(df_array, 'snr_db')
        df_avg = (
            df_clean.groupby(['snr_db', 'fs_hz'], observed=True)['doa_array_error_deg']
            .mean()
            .reset_index()
        )
        df_avg['fs_hz'] = pd.Categorical(df_avg['fs_hz'], ordered=True)
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=df_avg,
            x='snr_db',
            y='doa_array_error_deg',
            hue='fs_hz',
            style='fs_hz',
            markers=True,
            dashes=False,
            errorbar=None,
            palette='viridis'
        )
        plt.title("Error DOA vs SNR (por frecuencia)")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Error DOA (°)")
        plt.legend(title="Frecuencia (Hz)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

print("✅ Gráficos completos.")

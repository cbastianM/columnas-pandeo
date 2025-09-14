import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Columnas IPE", page_icon="🧮", layout="wide")
st.title("🧮 Diseño de Columnas IPE")
st.markdown("Desarrollado por:")
st.markdown("Sebastian Martínez Arrieta") 
st.markdown("Ana Victoria Pérez Ortega") 

PHI_C = 0.90  # φ fijo en compresión LRFD

def r3(x):
    try: return float(f"{x:.3f}")
    except Exception: return x

def need_cols(df, cols): return all(c in df.columns for c in cols)

# --- Entradas ---
with st.sidebar:
    st.header("Parámetros")
    Nd_kN = st.number_input("Carga de diseño N_d (kN)", min_value=0.0, value=800.0, step=10.0)
    L_m   = st.number_input("Longitud entre apoyos L (m)", min_value=0.1, value=3.0, step=0.1)
    K     = st.number_input("Coeficiente efectivo K (–)", min_value=0.1, value=1.0, step=0.05)
    E     = st.number_input("Módulo E (MPa)", min_value=100000.0, value=210000.0, step=1000.0)
    Fy    = st.number_input("Fy (MPa)", min_value=200.0, value=235.0, step=5.0)

st.markdown("### 📤 Archivo de perfiles")
archivo = st.file_uploader(
    "CSV o Excel con: Perfil, b (mm), tw (mm), tf (mm), area (cm2), d (mm), inercia_y (cm4), radio_y (cm), inercia_z (cm4), radio_z (cm)",
    type=["csv","xlsx"]
)
if not archivo:
    st.info("Sube el archivo para continuar."); st.stop()

df = pd.read_csv(archivo) if archivo.name.lower().endswith(".csv") else pd.read_excel(archivo)
cols_req = ["Perfil","b (mm)","tw (mm)","tf (mm)","area (cm2)","d (mm)","inercia_y (cm4)","radio_y (cm)","inercia_z (cm4)","radio_z (cm)"]
if not need_cols(df, cols_req):
    st.error(f"Faltan columnas. Deben estar todas: {', '.join(cols_req)}"); st.stop()

st.subheader("Datos cargados")
st.dataframe(df, use_container_width=True)

# --- Cálculos ---
Nd_N  = Nd_kN * 1e3
Le_mm = K * L_m * 1000.0

# Límites AISC automáticos
lamf_lim = 0.56 * np.sqrt(E / Fy)  # ala (usa b/2)
lamw_lim = 1.49 * np.sqrt(E / Fy)  # alma (d/tw)

rows = []
for _, row in df.iterrows():
    try:
        perfil = str(row["Perfil"])
        b  = float(row["b (mm)"])
        tw = float(row["tw (mm)"])
        tf = float(row["tf (mm)"])
        d  = float(row["d (mm)"])  # d es altura libre del alma
        A_cm2 = float(row["area (cm2)"])
        ry = float(row["radio_y (cm)"])*10.0
        rz = float(row["radio_z (cm)"])*10.0

        A_mm2 = A_cm2*100.0

        # --- Pandeo local ---
        c = b/2.0
        lam_f = c/tf if tf>0 else np.inf
        lam_w = d/tw if tw>0 else np.inf
        ok_local = (lam_f <= lamf_lim) and (lam_w <= lamw_lim)

        # --- Pandeo global (AISC E3) ---
        rmin = min(ry, rz)
        KLr = Le_mm/rmin if rmin>0 else np.inf
        Fe  = (np.pi**2 * E)/(KLr**2) if np.isfinite(KLr) and KLr>0 else 0.0
        lam_lim = 4.71*np.sqrt(E/Fy)
        Fcr = (0.658**(Fy/Fe))*Fy if (KLr<=lam_lim and Fe>0) else 0.877*Fe
        Pn = Fcr*A_mm2
        Pd = PHI_C*Pn
        util = Nd_N/Pd if Pd>0 else np.inf
        ok_global = Pd >= Nd_N
        ok_total = ok_local and ok_global

        rows.append({
            # Datos para optimización
            "_ok_total": ok_total,
            "_util": util,
            "Perfil": perfil,
            "Área (cm²)": r3(A_cm2),
            "λ_f": r3(lam_f),
            "Límite λ_f": r3(lamf_lim),
            "λ_w": r3(lam_w),
            "Límite λ_w": r3(lamw_lim),
            "KL/r": r3(KLr),
            "φPn (kN)": r3(Pd/1e3),
            "Nd (kN)": r3(Nd_kN),
            "Utilización": r3(util),
            "Local": "🟢" if ok_local else "🔴",
            "Flexión": "🟢" if ok_global else "🔴",
            "Total": "✅" if ok_total else "❌",
        })
    except Exception as e:
        st.warning(f"No se pudo calcular '{row.get('Perfil','?')}'. Error: {e}")

res_full = pd.DataFrame(rows)

# --- Selección óptima (no usa emojis) ---
cands = res_full[res_full["_ok_total"] == True].copy()
if len(cands) == 0:
    st.error("Ningún perfil cumple pandeo local y global con los parámetros dados."); st.stop()
cands = cands.sort_values(by=["Área (cm²)", "Utilización"], ascending=[True, True])
opt = cands.iloc[0]
st.success(f"**Perfil óptimo:** {opt['Perfil']}  |  Área = {opt['Área (cm²)']} cm²  |  φPn = {opt['φPn (kN)']} kN  |  Utilización = {opt['Utilización']}")

# --- Cuadro de texto con la explicación de columnas ---
st.markdown("""
- **λ_f**: esbeltez local del patín.  
- **Límite λ_f**: límite para patín.  
- **λ_w**: esbeltez local del alma.  
- **Límite λ_w**: límite para alma.  
- **KL/r**: esbeltez global de la columna.  
- **φPn (kN)**: resistencia por LRFD.  
- **Nd (kN)**: carga de diseño ingresada.  
- **Utilización**: Capacidad.  
""")



# --- Tabla resumida con solo cálculos importantes + emojis ---
st.subheader("Resultados por perfil (resumen)")
cols_show = ["Perfil","Área (cm²)","λ_f","Límite λ_f","λ_w","Límite λ_w","KL/r","φPn (kN)","Nd (kN)","Utilización","Local","Flexión","Total"]
st.dataframe(res_full[cols_show].reset_index(drop=True), use_container_width=True)

# -------- Detalle LaTeX del óptimo (sigue igual) --------
st.markdown("### 🧾 Fórmulas y cálculos")
row = df[df["Perfil"]==opt["Perfil"]].iloc[0]
b  = float(row["b (mm)"]); tw = float(row["tw (mm)"]); tf = float(row["tf (mm)"])
d  = float(row["d (mm)"]); A_cm2 = float(row["area (cm2)"])
ry = float(row["radio_y (cm)"])*10.0; rz = float(row["radio_z (cm)"])*10.0
A_mm2 = A_cm2*100.0
c=b/2.0
KLr = Le_mm/min(ry,rz)
Fe = (np.pi**2 * E)/(KLr**2)
lam_lim = 4.71*np.sqrt(E/Fy)
Fcr = (0.658**(Fy/Fe))*Fy if KLr<=lam_lim else 0.877*Fe
Pn = Fcr*A_mm2; Pd = PHI_C*Pn

b_s, c_s, tf_s, tw_s, d_s = map(lambda x: f"{r3(x)}",[b,c,tf,tw,d])
E_s, Fy_s = f"{r3(E)}", f"{r3(Fy)}"
lamf_lim_s, lamw_lim_s = f"{r3(lamf_lim)}", f"{r3(lamw_lim)}"
lamf_s, lamw_s = f"{r3(c/tf)}", f"{r3(d/tw)}"
Le_s, KLr_s = f"{r3(Le_mm)}", f"{r3(KLr)}"
Fe_s, Fcr_s, Pn_s, Pd_s, Nd_s = f"{r3(Fe)}", f"{r3(Fcr)}", f"{r3(Pn/1e3)}", f"{r3(Pd/1e3)}", f"{r3(Nd_kN)}"
lam_lim_s = f"{r3(lam_lim)}"

st.markdown("#### Pandeo local")
st.latex(rf"""\lambda_f=\frac{{b/2}}{{t_f}}=\frac{{{c_s}}}{{{tf_s}}}={lamf_s}\ \le\ 0.56\sqrt{{\frac{{E}}{{F_y}}}}=0.56\sqrt{{\frac{{{E_s}}}{{{Fy_s}}}}}={lamf_lim_s}""")
st.latex(rf"""\lambda_w=\frac{{h_w}}{{t_w}}=\frac{{d}}{{t_w}}=\frac{{{d_s}}}{{{tw_s}}}={lamw_s}\ \le\ 1.49\sqrt{{\frac{{E}}{{F_y}}}}=1.49\sqrt{{\frac{{{E_s}}}{{{Fy_s}}}}}={lamw_lim_s}""")

st.markdown("#### Pandeo flexionante")
st.latex(rf"""\lambda=\frac{{KL}}{{r_{{\min}}}}=\frac{{{r3(K)}\cdot {r3(L_m)}\cdot 1000}}{{r_{{\min}}}}={KLr_s}""")
st.latex(rf"""F_e=\frac{{\pi^2 E}}{{\lambda^2}}=\frac{{\pi^2\cdot {E_s}}}{{{KLr_s}^2}}={Fe_s}\ \text{{MPa}}""")
st.latex(rf"""\text{{Límite}}:\ 4.71\sqrt{{\frac{{E}}{{F_y}}}}=4.71\sqrt{{\frac{{{E_s}}}{{{Fy_s}}}}}={lam_lim_s}""")
if float(KLr_s) <= float(lam_lim_s):
    st.latex(rf"""F_{{cr}}=\left(0.658^{{F_y/F_e}}\right)F_y=\left(0.658^{{{Fy_s}/{Fe_s}}}\right)\,{Fy_s}={Fcr_s}\ \text{{MPa}}""")
else:
    st.latex(rf"""F_{{cr}}=0.877\,F_e=0.877\cdot {Fe_s}={Fcr_s}\ \text{{MPa}}""")
st.latex(rf"""P_n=F_{{cr}}A_g={Fcr_s}\cdot {r3(A_mm2)}={Pn_s}\ \text{{kN}}""")
st.latex(rf"""\phi=0.90,\ \phi P_n=0.90\cdot {Pn_s}={Pd_s}\ \text{{kN}},\ \eta=\frac{{N_d}}{{\phi P_n}}=\frac{{{Nd_s}}}{{{Pd_s}}}={r3(Nd_kN/(Pd/1e3))}""")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def simulate_and_display_meter(
    scenario_name, V_mag, V_ang, I_mag, pf_load, pf_type,
    v_terminal_map, i_terminal_map, i_polarity, pt_fuse_fail,
    ct_open, mode, simulation_hours
):
    phase_keys = ['R', 'Y', 'B']
    phase_colors = {'R': '#D32F2F', 'Y': "#E0B30F", 'B': '#1565C0'}

    # 1. Calculate True Source Quantities
    I_true_ang = []
    for k in range(3):
        phi = np.degrees(np.arccos(np.clip(pf_load[k], 0.0, 1.0)))
        I_true_ang.append(V_ang[k] - phi if pf_type[k].lower() == 'lag' else V_ang[k] + phi)

    system_voltages = {phase_keys[k]: (V_mag[k], V_ang[k]) for k in range(3)}
    system_currents = {phase_keys[k]: (I_mag[k], I_true_ang[k]) for k in range(3)}

    P_true_phases = [
        V_mag[k] * I_mag[k] * np.cos(np.radians(V_ang[k] - I_true_ang[k]))
        for k in range(3)
    ]
    P_true_total = sum(P_true_phases)

    # 2. Terminal Quantities Sampled by Meter DSP
    V_meter_mag, V_meter_ang = [], []
    I_meter_mag, I_meter_ang = [], []

    for k in range(3):
        vm, va = system_voltages[v_terminal_map[k]]
        if pt_fuse_fail[k]: vm = 0.0
        V_meter_mag.append(vm)
        V_meter_ang.append(va)

        im, ia = system_currents[i_terminal_map[k]]
        if ct_open[k]: im = 0.0
        if i_polarity[k] == -1:
            ia = (ia + 180.0) % 360.0
            if ia > 180.0: ia -= 360.0
        I_meter_mag.append(im)
        I_meter_ang.append(ia)

    # 3. DSP Vector Computations
    P_meas, Q_meas, S_meas, PF_meas, theta_deg_list = [], [], [], [], []
    q_powers = {'Q1': 0.0, 'Q2': 0.0, 'Q3': 0.0, 'Q4': 0.0}
    diagnostics = []

    for k in range(3):
        v = V_meter_mag[k]
        i = I_meter_mag[k]
        d_theta = (V_meter_ang[k] - I_meter_ang[k]) % 360.0
        if d_theta > 180.0: d_theta -= 360.0
        theta_deg_list.append(d_theta)
        
        rad = np.radians(d_theta)
        p = v * i * np.cos(rad)
        q = v * i * np.sin(rad)
        s = v * i
        pf = np.cos(rad) if s > 0 else 1.0

        P_meas.append(p)
        Q_meas.append(q)
        S_meas.append(s)
        PF_meas.append(pf)

        # Quadrant classification
        if p >= 0 and q >= 0: q_powers['Q1'] += q
        elif p < 0 and q >= 0: q_powers['Q2'] += q
        elif p < 0 and q < 0: q_powers['Q3'] += abs(q)
        elif p >= 0 and q < 0: q_powers['Q4'] += abs(q)

        # Diagnostics & Tamper Checks
        if pt_fuse_fail[k]: diagnostics.append(f"PT Potential Missing on Phase {phase_keys[k]}")
        if ct_open[k]: diagnostics.append(f"CT Circuit Open on Phase {phase_keys[k]}")
        if i_polarity[k] == -1: diagnostics.append(f"CT Polarity Inverted on Phase {phase_keys[k]}")
        if v_terminal_map[k] != phase_keys[k] or i_terminal_map[k] != phase_keys[k]:
            diagnostics.append(f"Phase Mismatch on Ch-{phase_keys[k]} (V:{v_terminal_map[k]}, I:{i_terminal_map[k]})")
        if p < -0.01 * s and s > 0:
            diagnostics.append(f"Reverse Power Flow on Phase {phase_keys[k]} ({p:.1f} W)")

    # 4. Total Meter Recorded Power & Energy
    if mode == "NET_VECTOR":
        P_meter_total = sum(P_meas)
    else:
        P_meter_total = sum(abs(p) for p in P_meas)

    error_pct = ((P_meter_total - P_true_total) / P_true_total * 100.0) if P_true_total != 0 else 0.0
    active_energy_kwh = (P_meter_total * simulation_hours) / 1000.0

    # 5. MATPLOTLIB DASHBOARD
    plt.close('all')
    fig = plt.figure(figsize=(19, 10), facecolor='#FFFFFF')
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.45], wspace=0.18, left=0.06, right=0.96, top=0.90, bottom=0.08)

    # LEFT PANEL: PHASOR DIAGRAM
    ax_polar = fig.add_subplot(gs[0, 0], projection='polar')
    ax_polar.set_facecolor('#FAFAFA')

    max_v = max(V_meter_mag) if max(V_meter_mag) > 0 else 1.0
    max_i = max(I_meter_mag) if max(I_meter_mag) > 0 else 1.0

    for k in range(3):
        pk = phase_keys[k]
        col = phase_colors[pk]

        if V_meter_mag[k] > 0:
            rad_v = np.radians(V_meter_ang[k])
            len_v = V_meter_mag[k] / max_v
            ax_polar.annotate('', xy=(rad_v, len_v), xytext=(0, 0),
                              arrowprops=dict(edgecolor=col, facecolor=col, arrowstyle='-|>', lw=2.8))
            ax_polar.text(rad_v, len_v * 1.22, f"V_{pk}\n{V_meter_mag[k]:.1f}V",
                          color=col, fontweight='bold', fontsize=9.5, ha='center', va='center')

        if I_meter_mag[k] > 0:
            rad_i = np.radians(I_meter_ang[k])
            len_i = (I_meter_mag[k] / max_i) * 0.72
            ax_polar.annotate('', xy=(rad_i, len_i), xytext=(0, 0),
                              arrowprops=dict(edgecolor=col, facecolor=col, arrowstyle='->', lw=2.2, ls='--'))
            ax_polar.text(rad_i, len_i * 1.25, f"I_{pk}\n{I_meter_mag[k]:.2f}A",
                          color=col, fontstyle='italic', fontweight='bold', fontsize=9.0, ha='center', va='center')

    ax_polar.set_theta_zero_location('E')
    ax_polar.set_theta_direction(1)
    ax_polar.set_title(f"METER VECTOR PHASOR DIAGRAM\n{scenario_name}\n",
                       fontsize=12, fontweight='bold', color='#0F172A', pad=15)
    ax_polar.grid(True, color='#CBD5E1', linestyle=':')
    ax_polar.set_yticklabels([])

    # RIGHT PANEL: DATA & TABLES
    ax_data = fig.add_subplot(gs[0, 1])
    ax_data.axis('off')

    headers = ["Phase", "V (V)", "∠V (°)", "I (A)", "∠I (°)", "Angle φ", "PF", "P (W)", "Q (var)", "S (VA)"]
    table_rows = []
    
    for k in range(3):
        pk = phase_keys[k]
        pf_flag = "Lag" if theta_deg_list[k] >= 0 else "Lead"
        table_rows.append([
            f"Phase {pk}", f"{V_meter_mag[k]:.2f}", f"{V_meter_ang[k]:.1f}°",
            f"{I_meter_mag[k]:.2f}", f"{I_meter_ang[k]:.1f}°", f"{theta_deg_list[k]:.1f}°",
            f"{PF_meas[k]:.3f} {pf_flag}", f"{P_meas[k]:.1f}", f"{Q_meas[k]:.1f}", f"{S_meas[k]:.1f}"
        ])

    col_widths = [0.11, 0.09, 0.10, 0.08, 0.10, 0.11, 0.14, 0.09, 0.09, 0.09]
    table = ax_data.table(cellText=[headers] + table_rows, colWidths=col_widths,
                          loc='upper left', cellLoc='center', bbox=[0.0, 0.68, 1.0, 0.30])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    for col_idx in range(len(headers)):
        c = table[(0, col_idx)]
        c.set_facecolor('#0F172A')
        c.set_text_props(color='#FFFFFF', fontweight='bold')
        c.set_height(0.08)

    for row_idx in range(1, 4):
        pk = phase_keys[row_idx - 1]
        for col_idx in range(len(headers)):
            c = table[(row_idx, col_idx)]
            c.set_edgecolor('#CBD5E1')
            c.set_height(0.07)
            if col_idx == 0:
                c.set_text_props(color=phase_colors[pk], fontweight='bold')

    sec1_y = 0.62
    ax_data.text(0.0, sec1_y, "METERING REGISTRATION & ACCURACY SUMMARY", 
                 fontweight='bold', fontsize=10.5, color='#0F172A')
    
    summary_text = (
        f"• Meter Logic Mode      : {mode}\n"
        f"• True Grid Power (P_in) : {P_true_total:.2f} W   |   Meter Recorded Power : {P_meter_total:.2f} W\n"
        f"• Registration Error    : {error_pct:+.2f} %  ({ 'Under-Recording' if error_pct < 0 else 'Over-Recording' })\n"
        f"• Active Energy Recorded: {active_energy_kwh:.4f} kWh  (Simulated interval: {simulation_hours:.1f} hr)"
    )
    ax_data.text(0.0, sec1_y - 0.03, summary_text, fontsize=9.5, fontfamily='monospace', 
                 va='top', color='#1E293B',
                 bbox=dict(boxstyle='square,pad=0.6', facecolor='#F8FAFC', edgecolor='#E2E8F0'))

    sec2_y = 0.42
    ax_data.text(0.0, sec2_y, "QUADRANT REACTIVE RECORDING (Instantaneous Rate & OBIS Codes)", 
                 fontweight='bold', fontsize=10.5, color='#0F172A')
    
    quadrant_text = (
        f"• Q1 (+P, +Q) [1.0.5.8.0.255] : {q_powers['Q1']:7.2f} var  ──  Inductive Import\n"
        f"• Q2 (-P, +Q) [1.0.6.8.0.255] : {q_powers['Q2']:7.2f} var  ──  Capacitive Export / Cross-Phase Reversed\n"
        f"• Q3 (-P, -Q) [1.0.7.8.0.255] : {q_powers['Q3']:7.2f} var  ──  Inductive Export\n"
        f"• Q4 (+P, -Q) [1.0.8.8.0.255] : {q_powers['Q4']:7.2f} var  ──  Capacitive Import"
    )
    ax_data.text(0.0, sec2_y - 0.03, quadrant_text, fontsize=9.5, fontfamily='monospace', 
                 va='top', color='#1E293B',
                 bbox=dict(boxstyle='square,pad=0.6', facecolor='#F8FAFC', edgecolor='#E2E8F0'))

    sec3_y = 0.20
    ax_data.text(0.0, sec3_y, "DIAGNOSTIC & TAMPER ANNUNCIATOR", 
                 fontweight='bold', fontsize=10.5, color='#B91C1C' if diagnostics else '#15803D')
    
    if diagnostics:
        alert_text = "\n".join([f" [!] ALERT: {d}" for d in set(diagnostics)])
    else:
        alert_text = " [✓] System Healthy. No connection or quantity abnormalities detected."
        
    ax_data.text(0.0, sec3_y - 0.03, alert_text, fontsize=9.5, fontfamily='monospace', 
                 va='top', color='#991B1B' if diagnostics else '#166534',
                 bbox=dict(boxstyle='square,pad=0.6', 
                           facecolor='#FEF2F2' if diagnostics else '#F0FDF4', 
                           edgecolor='#FECACA' if diagnostics else '#BBF7D0'))

    return fig

# ==============================================================================
# STREAMLIT UI SETUP
# ==============================================================================
st.set_page_config(page_title="Smart Meter Simulator", layout="wide")
st.title("⚡ Smart Meter Vector & Fault Simulator")

# --- MAIN DASHBOARD: EDITABLE GRID DATA ---
st.subheader("📝 Live Grid Configuration (Double click cells to edit)")

# Default Data for the Editable DataFrame
default_grid = pd.DataFrame({
    'Phase': ['R', 'Y', 'B'],
    'Voltage (V)': [63.51, 63.51, 63.51],
    'V_Angle (°)': [0.0, 120.0, 240.0],
    'Current (A)': [5.0, 5.0, 5.0],
    'PF': [0.85, 0.85, 0.85],
    'PF_Type': ['lag', 'lag', 'lag']
})

# Streamlit Data Editor
edited_grid = st.data_editor(
    default_grid,
    hide_index=True,
    use_container_width=True,
    column_config={
        "PF": st.column_config.NumberColumn(min_value=0.0, max_value=1.0),
        "PF_Type": st.column_config.SelectboxColumn(options=["lag", "lead"]),
        "Phase": st.column_config.TextColumn(disabled=True)
    }
)

# Extract edited lists
V_mags = edited_grid['Voltage (V)'].tolist()
V_angs = edited_grid['V_Angle (°)'].tolist()
I_mags = edited_grid['Current (A)'].tolist()
PFs = edited_grid['PF'].tolist()
PF_types = edited_grid['PF_Type'].tolist()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Simulation Settings")

scenario_name = st.sidebar.text_input("Scenario Name", "Custom Live Configuration")
mode = st.sidebar.selectbox("Meter Logic Mode", ["NET_VECTOR", "ABSOLUTE_FORWARD"])
sim_hours = st.sidebar.number_input("Simulation Hours", 0.1, 24.0, 1.0, 0.1)

st.sidebar.subheader("🔌 Phase Mapping (Terminals)")
v_map_R = st.sidebar.selectbox("R-Voltage maps to", ['R', 'Y', 'B'], index=0)
v_map_Y = st.sidebar.selectbox("Y-Voltage maps to", ['R', 'Y', 'B'], index=1)
v_map_B = st.sidebar.selectbox("B-Voltage maps to", ['R', 'Y', 'B'], index=2)

i_map_R = st.sidebar.selectbox("R-Current maps to", ['R', 'Y', 'B'], index=0)
i_map_Y = st.sidebar.selectbox("Y-Current maps to", ['R', 'Y', 'B'], index=1)
i_map_B = st.sidebar.selectbox("B-Current maps to", ['R', 'Y', 'B'], index=2)

st.sidebar.subheader("⚠️ Fault Injection")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown("**PT Fuse Fail**")
    pt_R = st.checkbox("Phase R PT blown")
    pt_Y = st.checkbox("Phase Y PT blown")
    pt_B = st.checkbox("Phase B PT blown")
with col2:
    st.markdown("**CT Open Circuit**")
    ct_R = st.checkbox("Phase R CT open")
    ct_Y = st.checkbox("Phase Y CT open")
    ct_B = st.checkbox("Phase B CT open")

st.sidebar.subheader("🔄 CT Polarity")
pol_R = st.radio("Phase R CT Polarity", [1, -1], index=0, horizontal=True)
pol_Y = st.radio("Phase Y CT Polarity", [1, -1], index=0, horizontal=True)
pol_B = st.radio("Phase B CT Polarity", [1, -1], index=0, horizontal=True)

# Compile configurations from UI
v_map = [v_map_R, v_map_Y, v_map_B]
i_map = [i_map_R, i_map_Y, i_map_B]
pt_faults = [pt_R, pt_Y, pt_B]
ct_faults = [ct_R, ct_Y, ct_B]
polarities = [pol_R, pol_Y, pol_B]

st.markdown("---")

# Generate and plot figure based on the editable grid and sidebar settings
fig = simulate_and_display_meter(
    scenario_name=scenario_name,
    V_mag=V_mags, V_ang=V_angs, I_mag=I_mags,
    pf_load=PFs, pf_type=PF_types,
    v_terminal_map=v_map, i_terminal_map=i_map,
    i_polarity=polarities, pt_fuse_fail=pt_faults,
    ct_open=ct_faults, mode=mode, simulation_hours=sim_hours
)

st.pyplot(fig)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def simulate_and_display_meter(
    scenario_name, V_mag, V_ang, I_mag, pf_load, pf_type,
    v_terminal_map, i_terminal_map, i_polarity, pt_fuse_fail,
    ct_open, mode, apparent_mode, simulation_hours
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
    P_true_total_kw = sum(P_true_phases) / 1000.0

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

    # 4. Total Meter Recorded Power & Energy Computations
    kva_import_sum = 0.0
    kva_export_sum = 0.0
    
    for idx, p in enumerate(P_meas):
        q = Q_meas[idx]
        s = S_meas[idx]
        
        # Apply Apparent Energy Mode Logic (Lag Only vs Lag+Lead)
        if apparent_mode == "Lag Only":
            # If leading (Q < 0), reactive power is ignored for apparent calculation (S = |P|)
            effective_s = s if q >= 0 else abs(p)
        else:
            effective_s = s
            
        if p >= 0:
            kva_import_sum += effective_s
        else:
            kva_export_sum += effective_s

    if mode == "NET_VECTOR":
        kw_import = sum(p for p in P_meas if p > 0) / 1000.0
        kw_export = sum(abs(p) for p in P_meas if p < 0) / 1000.0
        kva_import = kva_import_sum / 1000.0
        kva_export = kva_export_sum / 1000.0
    else: # ABSOLUTE_FORWARD
        kw_import = sum(abs(p) for p in P_meas) / 1000.0
        kw_export = 0.0
        kva_import = (kva_import_sum + kva_export_sum) / 1000.0
        kva_export = 0.0

    kvar_import = (q_powers['Q1'] + q_powers['Q4']) / 1000.0
    kvar_export = (q_powers['Q2'] + q_powers['Q3']) / 1000.0

    kwh_import, kwh_export = kw_import * simulation_hours, kw_export * simulation_hours
    kvarh_import, kvarh_export = kvar_import * simulation_hours, kvar_export * simulation_hours
    kvah_import, kvah_export = kva_import * simulation_hours, kva_export * simulation_hours

    # Accuracy Error based on Active Forward Registration vs True Forward Grid Power
    recorded_total_kw = kw_import if mode == "ABSOLUTE_FORWARD" else (kw_import - kw_export)
    error_pct = ((recorded_total_kw - P_true_total_kw) / P_true_total_kw * 100.0) if P_true_total_kw != 0 else 0.0

    # 5. MATPLOTLIB DASHBOARD
    plt.close('all')
    fig = plt.figure(figsize=(20, 11), facecolor='#F8FAFC')
    gs = GridSpec(1, 2, width_ratios=[0.9, 1.4], wspace=0.15, left=0.04, right=0.96, top=0.92, bottom=0.06)

    # ================= LEFT PANEL: PHASOR DIAGRAM =================
    ax_polar = fig.add_subplot(gs[0, 0], projection='polar')
    ax_polar.set_facecolor('#FFFFFF')

    max_v = max(V_meter_mag) if max(V_meter_mag) > 0 else 1.0
    max_i = max(I_meter_mag) if max(I_meter_mag) > 0 else 1.0

    for k in range(3):
        pk = phase_keys[k]
        col = phase_colors[pk]

        if V_meter_mag[k] > 0:
            rad_v = np.radians(V_meter_ang[k])
            len_v = V_meter_mag[k] / max_v
            ax_polar.annotate('', xy=(rad_v, len_v), xytext=(0, 0),
                              arrowprops=dict(edgecolor=col, facecolor=col, arrowstyle='-|>', lw=3.0))
            ax_polar.text(rad_v, len_v * 1.2, f"V_{pk}\n{V_meter_mag[k]:.1f}V",
                          color=col, fontweight='bold', fontsize=10, ha='center', va='center')

        if I_meter_mag[k] > 0:
            rad_i = np.radians(I_meter_ang[k])
            len_i = (I_meter_mag[k] / max_i) * 0.72
            ax_polar.annotate('', xy=(rad_i, len_i), xytext=(0, 0),
                              arrowprops=dict(edgecolor=col, facecolor=col, arrowstyle='->', lw=2.5, ls='--'))
            ax_polar.text(rad_i, len_i * 1.25, f"I_{pk}\n{I_meter_mag[k]:.2f}A",
                          color=col, fontstyle='italic', fontweight='bold', fontsize=9.5, ha='center', va='center')

    ax_polar.set_theta_zero_location('E')
    ax_polar.set_theta_direction(1)
    ax_polar.set_title(f"METER VECTOR PHASOR DIAGRAM\n{scenario_name}\n",
                       fontsize=13, fontweight='bold', color='#0F172A', pad=15)
    ax_polar.grid(True, color='#E2E8F0', linestyle='-')
    ax_polar.set_yticklabels([])
    ax_polar.spines['polar'].set_color('#CBD5E1')

    # ================= RIGHT PANEL: DATA & TABLES =================
    ax_data = fig.add_subplot(gs[0, 1])
    ax_data.axis('off')

    # --- TABLE 1: Instantaneous Phase Parameters ---
    headers_t1 = ["Phase", "V (V)", "∠V (°)", "I (A)", "∠I (°)", "Angle φ", "PF", "P (W)", "Q (var)", "S (VA)"]
    rows_t1 = []
    
    for k in range(3):
        pk = phase_keys[k]
        pf_flag = "Lag" if theta_deg_list[k] >= 0 else "Lead"
        rows_t1.append([
            f"Phase {pk}", f"{V_meter_mag[k]:.2f}", f"{V_meter_ang[k]:.1f}°",
            f"{I_meter_mag[k]:.2f}", f"{I_meter_ang[k]:.1f}°", f"{theta_deg_list[k]:.1f}°",
            f"{PF_meas[k]:.3f} {pf_flag}", f"{P_meas[k]:.1f}", f"{Q_meas[k]:.1f}", f"{S_meas[k]:.1f}"
        ])

    col_widths_t1 = [0.11, 0.09, 0.10, 0.08, 0.10, 0.10, 0.14, 0.10, 0.09, 0.09]
    t1 = ax_data.table(cellText=[headers_t1] + rows_t1, colWidths=col_widths_t1,
                       loc='upper left', cellLoc='center', bbox=[0.0, 0.74, 1.0, 0.26])
    
    t1.auto_set_font_size(False)
    t1.set_fontsize(10)

    for col_idx in range(len(headers_t1)):
        c = t1[(0, col_idx)]
        c.set_facecolor('#1E293B')
        c.set_text_props(color='#FFFFFF', fontweight='bold')
    
    for row_idx in range(1, 4):
        pk = phase_keys[row_idx - 1]
        for col_idx in range(len(headers_t1)):
            c = t1[(row_idx, col_idx)]
            c.set_edgecolor('#CBD5E1')
            c.set_facecolor('#FFFFFF')
            if col_idx == 0:
                c.set_text_props(color=phase_colors[pk], fontweight='bold')

    # --- TABLE 2: Registration & Energy Summary ---
    ax_data.text(0.0, 0.67, f"METER REGISTRATION SUMMARY  |  Logic Mode: {mode}  |  Apparent: {apparent_mode}  |  Sim: {simulation_hours} hr", 
                 fontweight='bold', fontsize=11, color='#0F172A')

    headers_t2 = ["Metric", "Import Power", "Export Power", "Import Energy", "Export Energy"]
    rows_t2 = [
        ["Active (kW / kWh)", f"{kw_import:.4f} kW", f"{kw_export:.4f} kW", f"{kwh_import:.4f} kWh", f"{kwh_export:.4f} kWh"],
        ["Reactive (kVAr / kVArh)", f"{kvar_import:.4f} kVAr", f"{kvar_export:.4f} kVAr", f"{kvarh_import:.4f} kVArh", f"{kvarh_export:.4f} kVArh"],
        ["Apparent (kVA / kVAh)", f"{kva_import:.4f} kVA", f"{kva_export:.4f} kVA", f"{kvah_import:.4f} kVAh", f"{kvah_export:.4f} kVAh"]
    ]

    col_widths_t2 = [0.24, 0.19, 0.19, 0.19, 0.19]
    t2 = ax_data.table(cellText=[headers_t2] + rows_t2, colWidths=col_widths_t2,
                       loc='upper left', cellLoc='center', bbox=[0.0, 0.40, 1.0, 0.25])
    
    t2.auto_set_font_size(False)
    t2.set_fontsize(10)

    for col_idx in range(len(headers_t2)):
        c = t2[(0, col_idx)]
        c.set_facecolor('#334155')
        c.set_text_props(color='#FFFFFF', fontweight='bold')

    for row_idx in range(1, 4):
        for col_idx in range(len(headers_t2)):
            c = t2[(row_idx, col_idx)]
            c.set_edgecolor('#CBD5E1')
            c.set_facecolor('#F8FAFC' if row_idx % 2 != 0 else '#FFFFFF')
            if col_idx == 0:
                c.set_text_props(fontweight='bold', color='#1E293B')

    # Accuracy text directly under Table 2
    err_color = '#15803D' if abs(error_pct) < 1.0 else '#B91C1C'
    ax_data.text(0.0, 0.35, f"True Grid Power: {P_true_total_kw:.4f} kW   |   Accuracy Error: {error_pct:+.2f}%", 
                 fontweight='bold', fontsize=11, color=err_color)

    # --- BLOCK 3: Quadrants ---
    ax_data.text(0.0, 0.28, "QUADRANT REACTIVE RECORDING (Instantaneous Rate & OBIS Codes)", 
                 fontweight='bold', fontsize=11, color='#0F172A')
    
    quadrant_text = (
        f"• Q1 (+P, +Q) [1.0.5.8.0.255] : {q_powers['Q1']/1000:7.4f} kVAr  ──  Inductive Import\n"
        f"• Q2 (-P, +Q) [1.0.6.8.0.255] : {q_powers['Q2']/1000:7.4f} kVAr  ──  Capacitive Export (or Reversed)\n"
        f"• Q3 (-P, -Q) [1.0.7.8.0.255] : {q_powers['Q3']/1000:7.4f} kVAr  ──  Inductive Export\n"
        f"• Q4 (+P, -Q) [1.0.8.8.0.255] : {q_powers['Q4']/1000:7.4f} kVAr  ──  Capacitive Import"
    )
    ax_data.text(0.0, 0.25, quadrant_text, fontsize=10, fontfamily='monospace', 
                 va='top', color='#334155', linespacing=1.8,
                 bbox=dict(boxstyle='square,pad=0.8', facecolor='#FFFFFF', edgecolor='#CBD5E1'))

    # --- BLOCK 4: Diagnostics ---
    ax_data.text(0.0, 0.08, "DIAGNOSTIC & TAMPER ANNUNCIATOR", 
                 fontweight='bold', fontsize=11, color='#B91C1C' if diagnostics else '#15803D')
    
    if diagnostics:
        alert_text = "\n".join([f" [!] ALERT: {d}" for d in set(diagnostics)])
    else:
        alert_text = " [✓] System Healthy. No connection or quantity abnormalities detected."
        
    ax_data.text(0.0, 0.05, alert_text, fontsize=10, fontfamily='monospace', 
                 va='top', color='#991B1B' if diagnostics else '#166534',
                 bbox=dict(boxstyle='square,pad=0.8', 
                           facecolor='#FEF2F2' if diagnostics else '#F0FDF4', 
                           edgecolor='#FECACA' if diagnostics else '#BBF7D0'))

    return fig

# ==============================================================================
# STREAMLIT UI SETUP
# ==============================================================================
st.set_page_config(page_title="Smart Meter Simulator", layout="wide")
st.title("⚡ Smart Meter Vector & Fault Simulator")

# --- TOP CONFIGURATION SECTION (Replaces Sidebar) ---
with st.expander("⚙️ Simulation & Hardware Settings", expanded=True):
    # Row 1: Global Settings[cite: 4]
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        scenario_name = st.text_input("Scenario Name", "Custom Live Configuration")
    with col_s2:
        mode = st.selectbox("Meter Logic Mode", ["NET_VECTOR", "ABSOLUTE_FORWARD"])
    with col_s3:
        apparent_mode = st.selectbox("Apparent Energy Logic", ["Lag+Lead", "Lag Only"])
    with col_s4:
        sim_hours = st.number_input("Simulation Hours", 0.1, 24.0, 1.0, 0.1)

    st.divider()

    # Row 2: Mapping and Hardware Faults[cite: 5, 2]
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.markdown("**🔌 Phase Mapping (Terminals)**")
        c1, c2 = st.columns(2)
        v_map_R = c1.selectbox("R-Volt Maps To", ['R', 'Y', 'B'], index=0)
        i_map_R = c2.selectbox("R-Curr Maps To", ['R', 'Y', 'B'], index=0)
        v_map_Y = c1.selectbox("Y-Volt Maps To", ['R', 'Y', 'B'], index=1)
        i_map_Y = c2.selectbox("Y-Curr Maps To", ['R', 'Y', 'B'], index=1)
        v_map_B = c1.selectbox("B-Volt Maps To", ['R', 'Y', 'B'], index=2)
        i_map_B = c2.selectbox("B-Curr Maps To", ['R', 'Y', 'B'], index=2)

    with col_m2:
        st.markdown("**⚠️ Fault Injection (Open Circuits)**")
        c3, c4 = st.columns(2)
        pt_R = c3.toggle("Phase R PT Blown")
        ct_R = c4.toggle("Phase R CT Open")
        pt_Y = c3.toggle("Phase Y PT Blown")
        ct_Y = c4.toggle("Phase Y CT Open")
        pt_B = c3.toggle("Phase B PT Blown")
        ct_B = c4.toggle("Phase B CT Open")

    with col_m3:
        st.markdown("**🔄 CT Polarity Reversal**")
        pol_R_toggle = st.toggle("Invert Phase R CT Polarity (-1)")
        pol_Y_toggle = st.toggle("Invert Phase Y CT Polarity (-1)")
        pol_B_toggle = st.toggle("Invert Phase B CT Polarity (-1)")
        
        # Convert toggle booleans to polarity integers
        pol_R = -1 if pol_R_toggle else 1
        pol_Y = -1 if pol_Y_toggle else 1
        pol_B = -1 if pol_B_toggle else 1

# Compile configurations from UI
v_map = [v_map_R, v_map_Y, v_map_B]
i_map = [i_map_R, i_map_Y, i_map_B]
pt_faults = [pt_R, pt_Y, pt_B]
ct_faults = [ct_R, ct_Y, ct_B]
polarities = [pol_R, pol_Y, pol_B]

# --- MAIN DASHBOARD: EDITABLE GRID DATA ---
st.subheader("📝 Live Grid Source Configuration (Double click cells to edit)")

default_grid = pd.DataFrame({
    'Phase': ['R', 'Y', 'B'],
    'Voltage (V)': [63.51, 63.51, 63.51],
    'V_Angle (°)': [0.0, 120.0, 240.0],
    'Current (A)': [5.0, 5.0, 5.0],
    'PF': [0.85, 0.85, 0.85],
    'PF_Type': ['lag', 'lag', 'lag']
})

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

V_mags = edited_grid['Voltage (V)'].tolist()
V_angs = edited_grid['V_Angle (°)'].tolist()
I_mags = edited_grid['Current (A)'].tolist()
PFs = edited_grid['PF'].tolist()
PF_types = edited_grid['PF_Type'].tolist()

st.markdown("---")

# Generate and plot figure based on the editable grid and top settings
fig = simulate_and_display_meter(
    scenario_name=scenario_name,
    V_mag=V_mags, V_ang=V_angs, I_mag=I_mags,
    pf_load=PFs, pf_type=PF_types,
    v_terminal_map=v_map, i_terminal_map=i_map,
    i_polarity=polarities, pt_fuse_fail=pt_faults,
    ct_open=ct_faults, mode=mode, apparent_mode=apparent_mode,
    simulation_hours=sim_hours
)

st.pyplot(fig)
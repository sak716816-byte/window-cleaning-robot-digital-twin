import os
import matplotlib.pyplot as plt

# Paths Setup
workspace_dir = r"C:\Users\16906\Desktop\中科绿洲"
formulas_dir = os.path.join(workspace_dir, "docs", "equations")
os.makedirs(formulas_dir, exist_ok=True)

def generate_formula_image(latex_str, filename, size=(6, 0.8), fontsize=14):
    path = os.path.join(formulas_dir, filename)
    
    # Create figure with a white background for perfect visibility in both GitHub light and dark modes
    fig = plt.figure(figsize=size, facecolor='white')
    
    # Render LaTeX
    plt.text(0.5, 0.5, f"${latex_str}$", size=fontsize, ha="center", va="center", color='black')
    plt.axis('off')
    
    # Save with white background (transparent=False) to ensure text isn't lost in dark mode
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor='white', edgecolor='none')
    plt.close()
    print(f"Generated: {path}")

# Math formulas in the README.md
formulas = {
    "eq_pressure.png": r"\frac{dP_i}{dt} = \frac{R \cdot T}{V_{cup}} \left( \dot{m}_{in, i} - \dot{m}_{out, i} \right)",
    "eq_leakage_flow.png": r"\dot{m}_{in, i} = C_d \cdot A_{leak, i} \cdot \sqrt{2\rho_{atm}(P_{atm} - P_i)}",
    "eq_leakage_area.png": r"A_{leak, i} = \pi \cdot D_{cup} \cdot \max(0, R_a - \delta_{seal}) + 2 \cdot w_{gap} \cdot d_{gap}",
    "eq_suction_flow.png": r"\dot{m}_{out, i} = \rho_i \cdot S_{max} \cdot \frac{\omega}{\omega_{max}} \cdot \frac{P_i - P_{limit}}{P_{atm} - P_{limit}}",
    "eq_normal_force.png": r"F_{normal} = F_{suction} + F_{preload} + F_{gravity, z} - F_{wind, normal} - F_{spray, normal}",
    "eq_slip_sf.png": r"SF_{slip} = \frac{\mu \cdot F_{normal}}{F_{tangential}}",
    "eq_overturn_sf.png": r"SF_{overturn} = \frac{M_{stabilizing}}{M_{overturning}}",
    "eq_mutex_flow.png": r"\text{SharedEnv} \rightarrow \text{Mutex.lock()} \rightarrow \text{Data Read/Write} \rightarrow \text{Mutex.unlock()}"
}

if __name__ == "__main__":
    for name, latex in formulas.items():
        # Adjust size for specific formulas if they are wider
        if name in ["eq_leakage_area.png", "eq_mutex_flow.png", "eq_suction_flow.png"]:
            generate_formula_image(latex, name, size=(8, 0.8))
        else:
            generate_formula_image(latex, name)
    print("All README formulas generated successfully!")

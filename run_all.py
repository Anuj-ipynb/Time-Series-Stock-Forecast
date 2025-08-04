import os
import subprocess
import time

# Ensure metrics directory exists
os.makedirs("metrics", exist_ok=True)

model_scripts = [
    "models/arima_model.py",
    "models/sarima_model.py",
    "models/prophet_model.py",
    "models/lstm_model.py"
]

print("📈 Running all forecasting models...\n")

for script in model_scripts:
    print(f"🚀 Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {script} completed successfully.")
    else:
        print(f"❌ {script} failed:\n{result.stderr}")

    print("-" * 50)
    time.sleep(1)

# Run seasonal decomposition
decomp_script = "models/decomposition_plot.py"
if os.path.exists(decomp_script):
    print("🌀 Running seasonal decomposition...")
    result = subprocess.run(["python", decomp_script], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Decomposition plot saved as decomposition_plot.png")
    else:
        print(f"❌ Decomposition failed:\n{result.stderr}")
else:
    print("⚠️ decomposition_plot.py not found. Skipping trend decomposition.")

print("\n🏁 All models executed. Forecasts and metrics are ready.")

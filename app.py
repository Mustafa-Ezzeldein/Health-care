import os
import joblib
import numpy as np
import pandas as pd
import threading
import time
import serial
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


models = {}


import __main__
import fuzzy_module
import clustering_module
__main__.FuzzyLogicSystem = fuzzy_module.FuzzyLogicSystem
__main__.ContinuousART = clustering_module.ContinuousART

try:
    __main__.FuzzyLogicSyystem = fuzzy_module.FuzzyLogicSystem
except:
    pass

# --- Arduino Serial Integration ---
SERIAL_PORT = 'COM7'
BAUD_RATE   = 9600

# Holds current + last valid readings
latest_sensor_data = {"bpm": 0, "spo2": 0, "temp": 0, "valid": False}
arduino_status = "Disconnected"

def read_from_arduino():
    global latest_sensor_data, arduino_status, ser_obj
    while True:
        try:
            print(f"📡 Attempting to connect to Arduino on {SERIAL_PORT}...")
            # Added dsrdtr=True which sometimes helps with certain USB-Serial chips
            ser_obj = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1, write_timeout=1)
            print(f"✅ Connected to {SERIAL_PORT}!")
            arduino_status = "Connected"
            time.sleep(2) # Wait for Arduino reboot after serial open
            while True:
                line = ser_obj.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"📥 Received: {line}")
                    if line.startswith("DATA:"):
                        arduino_status = "Connected"
                        # Format: DATA:HRate=75,Oxygen=98,Temp=37.2,VALID=1
                        payload = line[5:]
                        parts   = payload.split(",")
                        parsed  = {}
                        for part in parts:
                            if "=" in part:
                                key, val = part.split("=", 1)
                                parsed[key.strip()] = val.strip()

                        is_valid = parsed.get("VALID", "0") == "1"

                        try:
                            bpm  = float(parsed.get("HRate", 0))
                            spo2 = float(parsed.get("Oxygen", 0))
                            temp = float(parsed.get("Temp", 0))
                        except ValueError:
                            continue

                        # Always update temp so we can see raw values for debugging
                        latest_sensor_data["temp"] = round(temp, 1)

                        # Only update HR and SpO2 when the reading is declared valid
                        if is_valid:
                            if bpm  > 0: latest_sensor_data["bpm"]  = bpm
                            if spo2 > 0: latest_sensor_data["spo2"] = spo2
                            latest_sensor_data["valid"] = True
                        else:
                            # Finger not on sensor — keep last values but mark not-valid
                            latest_sensor_data["valid"] = False

        except Exception as e:
            arduino_status = "Disconnected"
            latest_sensor_data["valid"] = False
            print(f"❌ Serial Error: {e}. Retrying in 3 seconds...")
            time.sleep(3)

# Start Serial thread
threading.Thread(target=read_from_arduino, daemon=True).start()

def load_models():
    try: models['scaler'] = joblib.load('models/scaler.pkl')
    except Exception as e: print("⚠️ Scaler Error:", e)
        
    try: models['pca'] = joblib.load('models/pca_model.pkl')
    except Exception as e: print("⚠️ PCA Error:", e)

    try: models['rbf'] = joblib.load('models/rbf_model.pkl')
    except Exception as e: print("⚠️ RBF Error:", e)
        
    try: models['fuzzy'] = joblib.load('models/fuzzy_model.pkl')
    except Exception as e: print("⚠️ Fuzzy Error:", e)
            
    try: models['som'] = joblib.load('models/som_model.pkl')
    except Exception as e: print("⚠️ SOM Error:", e)

    try: models['art'] = joblib.load('models/art_model.pkl')
    except Exception as e: print("⚠️ ART Error:", e)

    try: models['rl'] = np.load('models/q_table.npy')
    except Exception as e: print("⚠️ RL Error:", e)

    try: models['gnn'] = np.load('models/gnn_adj.npy')
    except Exception as e: print("⚠️ GNN Error:", e)
    
    print("✅RBF Done!")

def generate_medication_advice(features, risk_score):
    advice = []
    # features: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]
    cp = features[2]
    trestbps = features[3]
    chol = features[4]
    fbs = features[5]
    
    if trestbps > 140:
        advice.append({"type": "Blood Pressure", "med": "Beta-Blockers / ACE Inhibitors", "desc": "High blood pressure detected."})
    elif trestbps < 90:
        advice.append({"type": "Blood Pressure", "med": "Fluids / Midodrine", "desc": "Low blood pressure (Hypotension) detected."})
    
    if chol > 240:
        advice.append({"type": "Cholesterol", "med": "High-intensity Statins", "desc": "Significantly elevated cholesterol."})
    elif chol > 200:
        advice.append({"type": "Cholesterol", "med": "Statins / Omega-3", "desc": "Borderline high cholesterol level."})
    
    # New Mapping: 0: Healthy, 1: Non-anginal, 2: Atypical, 3: Typical Angina
    if cp >= 2:
        advice.append({"type": "Chest Pain", "med": "Nitroglycerin / Aspirin", "desc": "Patterns of angina detected."})
        
    if fbs > 1: # Usually fbs is binary (0/1) in heart data, but sometimes >120 mg/dl is used
        advice.append({"type": "Blood Sugar", "med": "Metformin / Insulin", "desc": "High fasting blood sugar."})
    elif fbs == 1:
        advice.append({"type": "Blood Sugar", "med": "Dietary Management", "desc": "Fasting blood sugar > 120mg/dl."})
        
    if risk_score > 75:
        advice.append({"type": "Urgent", "med": "Immediate Hospitalization", "desc": "Critical heart risk detected."})
    
    return advice

load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_sensors')
def get_sensors():
    data = latest_sensor_data.copy()
    data['status'] = arduino_status
    return jsonify(data)

@app.route('/control_arduino')
def control_arduino():
    cmd = request.args.get('cmd', '').upper()
    try:
        global ser_obj, latest_sensor_data
        # Clear cache so it doesn't instantly load old readings on page refresh or reset
        if cmd == 'ON':
            latest_sensor_data = {"bpm": 0, "spo2": 0, "temp": 0, "valid": False}
            
        if 'ser_obj' in globals() and ser_obj and ser_obj.is_open:
            ser_obj.write(f"{cmd}\n".encode())
            return jsonify({"status": "success", "command": cmd})
        return jsonify({"status": "error", "message": "Serial not connected"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
 
        features = [
            float(data.get('age', 50)),
            float(data.get('sex', 1)),
            float(data.get('cp', 0)),
            float(data.get('trestbps', 120)),
            float(data.get('chol', 200)),
            float(data.get('fbs', 0)),
            float(data.get('restecg', 1)),
            float(data.get('thalach', 150)),
            float(data.get('exang', 0)),
            float(data.get('oldpeak', 0.0)),
            float(data.get('slope', 1)),
            float(data.get('oxygen', 98)),
            float(data.get('temp', 37))
        ]
        
        fuzzy_risk = 0.0
        if 'fuzzy' in models:
            fz_risks = models['fuzzy'].predict([features])
            fuzzy_risk = float(fz_risks[0])
            
        
        rbf_risk = 0.0
        if 'scaler' in models and 'pca' in models and 'rbf' in models:
            
            cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'oxygen', 'temp']
            x_raw = pd.DataFrame([features], columns=cols)
            x_scaled = models['scaler'].transform(x_raw)
            x_pca = models['pca'].transform(x_scaled)
            x_pca_df = pd.DataFrame(x_pca, columns=[f'PC{i+1}' for i in range(x_pca.shape[1])])
            
           
            try:
                rbf_risk = float(models['rbf'].predict_proba(x_pca_df)[0][1] * 100)
            except:
                pred = models['rbf'].predict(x_pca_df)[0]
                rbf_risk = 85.0 if pred == 1 else 15.0

   
        aggregated_risk = max(fuzzy_risk, rbf_risk)

        # 3. RL Treatment Action 🚑 (Reinforcement Learning)
        rl_action = "N/A"
        if 'rl' in models:
            if aggregated_risk < 33: state = 0       # Low Risk
            elif aggregated_risk < 66: state = 1     # Medium Risk
            else: state = 2                     # High Risk
            
            q_table = models['rl']
            best_action_idx = np.argmax(q_table[state])
            actions_desc = ["Routine Check ✅", "Alert Doctor ⚠️", "Emergency Call 🚑🚨"]
            rl_action = actions_desc[best_action_idx]

        # 4. Unsupervised Clustering (SOM & ART2) 🧬
        som_status = "N/A"
        art_status = "N/A"
        
        if 'som' in models and 'pca' in models:
            winner_node = models['som'].winner(x_pca[0])
            
            if aggregated_risk < 35: som_status = "Healthy"
            elif aggregated_risk < 70: som_status = "At Risk"
            else: som_status = "Critical"

       
        request_vig = float(data.get('vigilance', 0.82))
        
        if 'art' in models and 'pca' in models:
            art_cluster = -1
            x_input = x_pca[0] 
            
            for i, c in enumerate(models['art'].clusters):
                sim = np.dot(x_input, c) / (np.linalg.norm(x_input) * np.linalg.norm(c) + 1e-8)
                if sim >= request_vig:
                    art_cluster = i
                    break
            
            
            if art_cluster == -1: 
                art_status = "New Pattern Detected ⚠️"
            else:
                art_risk_score = aggregated_risk
                if art_risk_score < 35: art_status = "Healthy"
                elif art_risk_score < 70: art_status = "At Risk"
                else: art_status = "Critical"

        # 5. GNN Biological Network Visuals 🕸️ - Force Interconnectivity
        gnn_edges = []
       
        nodes = [0, 2, 3, 7, 11, 12]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                gnn_edges.append({'source': nodes[i], 'target': nodes[j]})

        # 6. Medication Suggestions 💊
        medications = generate_medication_advice(features, aggregated_risk)

        return jsonify({
            'fuzzy_risk': fuzzy_risk,
            'rbf_risk': rbf_risk,
            'rl_action': rl_action,
            'som_status': som_status,
            'art_status': art_status,
            'gnn_edges': gnn_edges,
            'medications': medications,
            'status': 'success'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

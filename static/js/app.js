const inputs = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'oxygen', 'temp'];

function updateVal(inputId, valId) {
    const el = document.getElementById(inputId);
    const target = document.getElementById(valId);
    if (!el || !target) return;
    
    let val = el.value;
    if (inputId === "sex") target.innerText = (val == "1" ? "1 (Male)" : "0 (Female)");
    else if (inputId === "cp") {
        const types = ["No Pain", "Non-anginal", "Atypical Angina", "Typical Angina"];
        target.innerText = val + " (" + types[val] + ")";
    }
    else if (inputId === "fbs") target.innerText = (val == "1" ? "1 (High)" : "0 (Normal)");
    else if (inputId === "exang") target.innerText = (val == "1" ? "1 (Yes)" : "0 (No)");
    else target.innerText = val;
}

async function analyzePatient() {
    const btn = document.getElementById('analyzeBtn');
    if(btn) btn.innerText = "Processing Neural Data...";

    const data = {};
    inputs.forEach(id => {
        const el = document.getElementById(id);
        if(el) data[id] = el.value;
    });

    const vig = document.getElementById('vigSlider');
    if(vig) data['vigilance'] = parseFloat(vig.value);

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        const result = await res.json();
        
        if (result.status === 'success') {
            updateBar('rbfBar', 'rbfText', result.rbf_risk);
            updateBar('fuzzyBar', 'fuzzyText', result.fuzzy_risk);
            
            // Update Decision Card (RL Alert) and side colors
            const rl = document.getElementById('rlAlert');
            if(rl) {
                rl.innerText = result.rl_action;
                const card = rl.closest('.ai-card'); // Get the whole box
                
                if(result.rl_action.includes('Emergency') || result.rl_action.includes('🚨')) {
                    rl.style.color = '#ff0055'; 
                    rl.style.borderColor = '#ff0055';
                    rl.style.backgroundColor = 'rgba(255,0,85,0.1)';
                    card.style.borderLeftColor = '#ff0055'; // Red bar
                    card.style.boxShadow = '0 0 20px rgba(255, 0, 85, 0.2)';
                } else if(result.rl_action.includes('Routine') || result.rl_action.includes('✅')) {
                    rl.style.color = '#00ff88'; 
                    rl.style.borderColor = '#00ff88';
                    rl.style.backgroundColor = 'rgba(0,255,136,0.1)';
                    card.style.borderLeftColor = '#00ff88'; // Green bar
                    card.style.boxShadow = 'none';
                } else {
                    rl.style.color = '#ffcc00'; 
                    rl.style.borderColor = '#ffcc00';
                    rl.style.backgroundColor = 'rgba(255,204,0,0.1)';
                    card.style.borderLeftColor = '#ffcc00'; // Yellow bar
                    card.style.boxShadow = 'none';
                }
            }
            
            updateStatus('somText', result.som_status);
            updateStatus('artText', result.art_status);

            const medList = document.getElementById('medList');
            if(medList) {
                medList.innerHTML = '';
                result.medications.forEach(m => {
                    const d = document.createElement('div');
                    d.className = 'med-item';
                    d.innerHTML = `<strong>${m.type}: ${m.med}</strong><span>${m.desc}</span>`;
                    medList.appendChild(d);
                });
            }

            window.lastGNNDATA = { edges: result.gnn_edges, risks: result.fuzzy_risk };
        }
    } catch (e) {
        console.error("Prediction Error:", e);
    } finally {
        if(btn) btn.innerText = "Run AI Analysis";
    }
}

function updateBar(barId, textId, value) {
    const bar = document.getElementById(barId);
    const text = document.getElementById(textId);
    if(bar && text) {
        bar.style.width = value + '%';
        text.innerText = value.toFixed(1) + '%';
        bar.style.backgroundColor = value > 66 ? '#ff0055' : '#00f0ff';
    }
}

function updateStatus(id, text) {
    const el = document.getElementById(id);
    if(!el) return;
    el.innerText = text;
    if(text === 'Critical' || text.includes('New')) el.style.color = '#ff0055';
    else el.style.color = '#00f0ff';
}

// --- GNN Visualization ---
const canvas = document.getElementById('gnnCanvas');
const ctx = canvas.getContext('2d');

const nodePositions = {
    0: {x: 0.5, y: 0.35, label: "AGE/SEX"},  
    2: {x: 0.42, y: 0.48, label: "CHEST PAIN"}, 
    3: {x: 0.55, y: 0.62, label: "BP"}, 
    7: {x: 0.62, y: 0.45, label: "HEART RATE"}, 
    11: {x: 0.48, y: 0.52, label: "OXYGEN"}, 
    12: {x: 0.52, y: 0.72, label: "TEMP"}
};

function drawGNN() {
    if(!window.lastGNNDATA || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    if(canvas.width !== rect.width) {
        canvas.width = rect.width;
        canvas.height = rect.height;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const risk = window.lastGNNDATA.risks;
    const isCritical = risk > 66;
    const mainColor = isCritical ? '#ff0055' : '#00f0ff';
    const time = Date.now() * 0.002;
    const edges = window.lastGNNDATA.edges;

    edges.forEach(edge => {
        const start = nodePositions[edge.source];
        const end = nodePositions[edge.target];
        if(!start || !end) return;
        ctx.beginPath();
        ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
        ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
        ctx.strokeStyle = mainColor + "55";
        ctx.lineWidth = 3;
        ctx.stroke();
        const pulsePos = (time % 1);
        ctx.beginPath();
        ctx.arc((start.x + (end.x - start.x) * pulsePos) * canvas.width, (start.y + (end.y - start.y) * pulsePos) * canvas.height, 3, 0, Math.PI*2);
        ctx.fillStyle = "#fff"; ctx.fill();
    });

    ctx.font = "bold 9px 'Orbitron', sans-serif";
    ctx.textAlign = "center";
    Object.keys(nodePositions).forEach(key => {
        const p = nodePositions[key];
        const x = p.x * canvas.width;
        const y = p.y * canvas.height;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI*2);
        ctx.fillStyle = mainColor; ctx.fill();
        ctx.fillStyle = "#fff";
        ctx.fillText(p.label, x, y - 12);
    });
}
setInterval(drawGNN, 40);

// --- Arduino Live Connection ---
let arduinoInterval = null;
let readingLocked   = false;   // true = we have a valid capture, sliders frozen

async function checkArduinoConnection() {
    const btn    = document.getElementById('arduinoBtn');
    const status = document.getElementById('arduinoStatus');
    if (!btn) return;
    try {
        const res  = await fetch('/get_sensors');
        const data = await res.json();
        if (data.status === 'Connected') {
            btn.disabled      = false;
            btn.style.opacity = '1';
            btn.style.cursor  = 'pointer';
            if (!arduinoInterval) {
                status.innerText    = 'Arduino Ready ✅ — Click Connect';
                status.style.color  = '#00ff88';
            }
        } else {
            btn.disabled      = false;
            btn.style.opacity = '0.6';
            btn.style.cursor  = 'pointer';
            if (!arduinoInterval) {
                status.innerText   = 'Waiting for Arduino... ⏳';
                status.style.color = '#ffcc00';
            }
            // If polling was active and Arduino disconnected, stop
            if (arduinoInterval && !readingLocked) {
                clearInterval(arduinoInterval);
                arduinoInterval = null;
                resetArduinoBtn();
            }
        }
    } catch(e) {
        if (btn) btn.disabled = false;
    }
}
setInterval(checkArduinoConnection, 2000);

function resetArduinoBtn() {
    const btn = document.getElementById('arduinoBtn');
    if (!btn) return;
    btn.classList.remove('active');
    btn.innerHTML          = `<i class="fa-solid fa-microchip"></i> Connect Arduino`;
    btn.style.backgroundColor = 'transparent';
    btn.style.color           = '#00f0ff';
}

function unlockReading() {
    readingLocked = false;
    
    // Clear display and show searching
    const status = document.getElementById('arduinoStatus');
    if (status) { status.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Resetting... Please lift finger for a moment.`; status.style.color = '#ffcc00'; }
    
    const btn = document.getElementById('arduinoBtn');
    if (btn) {
        btn.innerHTML = `<i class="fa-solid fa-microchip"></i> Searching...`;
        btn.style.backgroundColor = '#ffcc00';
        btn.style.color = '#000';
    }

    ['thalach','oxygen','temp'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.disabled = false;
            el.value = '';
            updateVal(id, id + 'Val'); // trigger visual reset
        }
    });

    // Send reset command to backend to clear old buffers if needed
    fetch('/control_arduino?cmd=ON').catch(e => {});

    // Add a 2.5 second ignore window so it doesn't instantly relock if finger is still there
    window.ignoreReadingsUntil = Date.now() + 2500;

    setTimeout(() => {
        if (!readingLocked && status) {
            status.innerText = '👆 Ready for new reading';
        }
    }, 2500);
}


async function stopArduino() {
    // Send OFF command to Arduino
    try { await fetch('/control_arduino?cmd=OFF'); } catch(e) {}
    
    document.getElementById('closeBtn').style.display = 'block';
    if (arduinoInterval) {
        clearInterval(arduinoInterval);
        arduinoInterval = null;
    }
    readingLocked = false;
    resetArduinoBtn();
    
    const status = document.getElementById('arduinoStatus');
    if (status) { 
        status.innerHTML = '<span style="color:#ff0055">🔴 Sensor Closed & LED Off</span><br><button onclick="toggleArduino()" style="margin-top:5px;padding:2px 10px;background:#00f0ff;color:#000;border:none;border-radius:4px;cursor:pointer;">Re-Open Sensor</button>'; 
    }
    document.getElementById('closeBtn').style.display = 'none';
}

function toggleArduino() {
    const btn = document.getElementById('arduinoBtn');
    if (arduinoInterval) {
        stopArduino();
        return;
    }

    // Start polling & Turn ON LED
    fetch('/control_arduino?cmd=ON').catch(e => {});
    document.getElementById('closeBtn').style.display = 'block';
    
    readingLocked = false;
    btn.classList.add('active');
    btn.innerHTML             = `<i class="fa-solid fa-microchip"></i> Searching...`;
    btn.style.backgroundColor = '#ffcc00';
    btn.style.color           = '#000';

    arduinoInterval = setInterval(async () => {
        if (readingLocked) return;

        try {
            const res  = await fetch('/get_sensors');
            const data = await res.json();
            const status = document.getElementById('arduinoStatus');

            if (data.status !== 'Connected') {
                if (status) { status.innerText = 'Arduino disconnected ❌'; status.style.color = '#ff0055'; }
                return;
            }

            // Always show live temp (updates even if 0)
            if (data.temp !== undefined) {
                const el = document.getElementById('temp');
                if (el) { el.value = data.temp.toFixed(1); updateVal('temp', 'tempVal'); }
            }

            // 🟢 LIVE FEED: Show values even before they are 'VALID' so user knows it's working
            if (data.bpm > 0) {
                const hrEl = document.getElementById('thalach');
                if (hrEl) { hrEl.value = data.bpm; updateVal('thalach', 'thalachVal'); }
            }
            if (data.spo2 > 0) {
                const o2El = document.getElementById('oxygen');
                if (o2El) { o2El.value = data.spo2; updateVal('oxygen', 'oxygenVal'); }
            }

            // If VALID reading → lock everything
            if (data.valid && data.bpm > 30 && data.spo2 > 70) {
                if (Date.now() < window.ignoreReadingsUntil) {
                    // Still in cooldown period after reset, just show live data but don't lock
                    if (status) {
                        status.innerHTML = `⏳ Please lift finger to reset...`;
                        status.style.color = '#ffcc00';
                    }
                    return; // Skip locking
                }

                readingLocked = true;
                
                ['thalach','oxygen','temp'].forEach(id => {
                    const el = document.getElementById(id);
                    if (el) el.disabled = true;
                });

                const btn = document.getElementById('arduinoBtn');
                btn.innerHTML             = `<i class="fa-solid fa-lock"></i> Reading Locked`;
                btn.style.backgroundColor = '#00ff88';
                btn.style.color           = '#000';

                if (status) {
                    status.innerHTML = `
                        <div style="background: rgba(0, 255, 136, 0.1); border: 1px solid #00ff88; padding: 10px; border-radius: 8px; margin-top: 10px;">
                            🎯 Captured: BPM=<b>${data.bpm}</b> | SpO2=<b>${data.spo2}%</b> | Temp=<b>${data.temp}°C</b>
                        </div>
                        <button onclick="unlockReading()" style="margin-top:15px; width:100%; padding:10px; background:#ffcc00; color:#000; border:none; border-radius:8px; font-weight:bold; cursor:pointer; box-shadow: 0 4px 15px rgba(255,204,0,0.3);">
                            <i class="fa-solid fa-rotate-right"></i> Reset & Measure Again
                        </button>`;
                    status.style.color = '#00ff88';
                }
            } else {
                // Finger not on sensor yet
                if (status) {
                    const ir_hint = data.bpm === 0 ? '👆 Place finger on sensor' : '⏳ Stabilizing...';
                    status.innerText   = ir_hint;
                    status.style.color = '#ffcc00';
                    const btn = document.getElementById('arduinoBtn');
                    btn.innerHTML      = `<i class="fa-solid fa-microchip"></i> Waiting for finger...`;
                    btn.style.backgroundColor = '#ffcc00';
                    btn.style.color           = '#000';
                }
            }
        } catch(e) { console.error('Sensor fetch error:', e); }
    }, 300);
}


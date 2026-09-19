// ============================================================
//  connect_backend.js
//  ADD THIS CODE to SignRecognitionApp_FINAL.html
//  Connects HTML app to Python FastAPI backend
// ============================================================
//
//  INSTRUCTIONS:
//  1. Open SignRecognitionApp_FINAL.html in VS Code
//  2. Find this line near the bottom:  </script>
//  3. PASTE all code below JUST BEFORE that </script> line
//  4. Save the file
//  5. Start server.py first, then open HTML in Chrome

// ── BACKEND CONNECTION ──────────────────────────────────────
let ws = null;
let wsConnected = false;
let backendCanvas = null;
let backendCtx = null;
let frameInterval = null;

// Try to connect to Python backend
function connectBackend() {
    try {
        ws = new WebSocket('ws://localhost:8000/ws');

        ws.onopen = () => {
            wsConnected = true;
            console.log('✅ Connected to Python backend!');
            showBackendStatus(true);
            startSendingFrames();
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'pong') return;
            handleBackendResult(data);
        };

        ws.onclose = () => {
            wsConnected = false;
            console.log('🔴 Backend disconnected - using browser detection');
            showBackendStatus(false);
            stopSendingFrames();
            // Retry connection after 3 seconds
            setTimeout(connectBackend, 3000);
        };

        ws.onerror = () => {
            wsConnected = false;
            showBackendStatus(false);
        };

    } catch(e) {
        console.log('Backend not available - using browser only mode');
    }
}

// Show connection status in the UI
function showBackendStatus(connected) {
    // Try to find existing status element or create one
    let statusEl = document.getElementById('backend-status');
    if (!statusEl) {
        statusEl = document.createElement('div');
        statusEl.id = 'backend-status';
        statusEl.style.cssText = `
            position: fixed;
            top: 12px;
            right: 12px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: bold;
            z-index: 9999;
            font-family: Arial;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        `;
        document.body.appendChild(statusEl);
    }

    if (connected) {
        statusEl.textContent = '🟢 Python Backend Connected';
        statusEl.style.background = '#064e3b';
        statusEl.style.color = '#34d399';
        statusEl.style.border = '1px solid #34d399';
    } else {
        statusEl.textContent = '🟡 Browser-Only Mode';
        statusEl.style.background = '#1c1917';
        statusEl.style.color = '#fbbf24';
        statusEl.style.border = '1px solid #fbbf24';
    }
}

// Send video frames to Python backend
function startSendingFrames() {
    // Create hidden canvas for frame capture
    if (!backendCanvas) {
        backendCanvas = document.createElement('canvas');
        backendCanvas.width  = 320;  // Small size = faster transfer
        backendCanvas.height = 240;
        backendCtx = backendCanvas.getContext('2d');
    }

    stopSendingFrames(); // Clear any existing interval

    frameInterval = setInterval(() => {
        if (!wsConnected || ws.readyState !== WebSocket.OPEN) return;

        // Find the video element on the page
        const video = document.querySelector('video');
        if (!video || video.paused || video.ended) return;

        try {
            // Draw current video frame to canvas
            backendCtx.drawImage(video, 0, 0, 320, 240);

            // Send as base64 JPEG (compressed for speed)
            const frameData = backendCanvas.toDataURL('image/jpeg', 0.6);
            ws.send(JSON.stringify({
                type:  'frame',
                frame: frameData
            }));
        } catch(e) {
            // Ignore frame errors
        }
    }, 120); // Send ~8 frames per second to backend
}

function stopSendingFrames() {
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }
}

// Handle result from Python backend
function handleBackendResult(data) {
    if (!data.detected) return;

    // Only process confirmed signs (stable detection)
    if (!data.confirmed) return;

    const signEN = data.confirmed_sign;
    const signTA = data.confirmed_tamil;
    const conf   = data.confidence;

    console.log(`✅ Backend: ${signEN} (${signTA}) - ${conf}%`);

    // ── Update the HTML app UI ──────────────────────────────
    // Try to update the English output box
    const enBox = document.getElementById('outputEnglish') ||
                  document.querySelector('[id*="nglish"]') ||
                  document.querySelector('[id*="output"]');
    if (enBox) enBox.value = signEN;

    // Try to update Tamil output box
    const taBox = document.getElementById('outputTamil') ||
                  document.querySelector('[id*="amil"]');
    if (taBox) taBox.value = signTA;

    // Try to update any detection display
    const detectionEl = document.querySelector('.detection-result') ||
                        document.querySelector('[class*="result"]') ||
                        document.querySelector('[class*="detect"]');
    if (detectionEl) {
        detectionEl.textContent = `${signEN} — ${signTA}`;
    }

    // Show a floating result popup
    showFloatingResult(signEN, signTA, conf);

    // Auto-speak if TTS is available
    autoSpeak(signEN);
}

// Show floating result on screen
let floatingEl = null;
function showFloatingResult(signEN, signTA, conf) {
    if (!floatingEl) {
        floatingEl = document.createElement('div');
        floatingEl.style.cssText = `
            position: fixed;
            bottom: 100px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(10, 25, 47, 0.95);
            border: 2px solid #38bdf8;
            border-radius: 16px;
            padding: 16px 32px;
            text-align: center;
            z-index: 9998;
            font-family: Arial, 'Noto Sans Tamil', sans-serif;
            box-shadow: 0 8px 32px rgba(56,189,248,0.3);
            transition: opacity 0.3s;
            min-width: 260px;
        `;
        document.body.appendChild(floatingEl);
    }

    floatingEl.innerHTML = `
        <div style="font-size:28px;font-weight:bold;color:#38bdf8;
                    letter-spacing:2px;">${signEN}</div>
        <div style="font-size:22px;color:#fbbf24;margin-top:6px;
                    font-family:'Noto Sans Tamil',Arial,sans-serif;">
            ${signTA}
        </div>
        <div style="font-size:12px;color:#64748b;margin-top:4px;">
            ${conf}% confidence • Python Backend
        </div>
    `;
    floatingEl.style.opacity = '1';

    // Fade out after 2.5 seconds
    clearTimeout(floatingEl._timer);
    floatingEl._timer = setTimeout(() => {
        floatingEl.style.opacity = '0';
    }, 2500);
}

// Auto speak the detected sign
let lastSpoken = '';
let speakCooldown = false;

function autoSpeak(text) {
    if (speakCooldown || text === lastSpoken) return;
    if (!window.speechSynthesis) return;

    lastSpoken = text;
    speakCooldown = true;
    setTimeout(() => { speakCooldown = false; }, 2000);

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.pitch  = 2.0;   // Shinchan style!
    utterance.rate   = 1.2;
    utterance.volume = 1.0;
    window.speechSynthesis.speak(utterance);
}

// ── START CONNECTION WHEN PAGE LOADS ─────────────────────────
window.addEventListener('load', () => {
    console.log('🚀 Connecting to Python backend...');
    showBackendStatus(false);
    connectBackend();
});

// Reconnect when camera starts (user clicks Start Camera)
document.addEventListener('click', (e) => {
    if (e.target.textContent && 
        (e.target.textContent.includes('Start') || 
         e.target.textContent.includes('Camera'))) {
        if (!wsConnected) connectBackend();
        setTimeout(startSendingFrames, 1000);
    }
});

console.log('✅ Backend connector loaded!');

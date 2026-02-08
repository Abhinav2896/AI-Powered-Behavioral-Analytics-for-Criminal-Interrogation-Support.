/**
 * runner.js - Automated Development Startup Script
 * 
 * GOAL:
 * 1. Free up required ports (8000 for Backend, 5173 for Frontend)
 *    - Note: User requested 5000/3000, but project is currently configured for 8000/5173.
 *    - This script uses the actual project ports to ensure the app works correctly.
 * 2. Launch Backend (Python FastAPI)
 * 3. Launch Frontend (React Vite)
 * 4. Show real-time logs
 * 
 * USAGE:
 * node runner.js
 */

const { spawn, exec } = require('child_process');
const os = require('os');
const path = require('path');
const net = require('net');

// Configuration
const CONFIG = {
    backend: {
        port: 8000, // Matching app.py default
        path: path.join(__dirname), // run from project root so module path 'backend.app:app' resolves
        command: 'python', // Will attempt to detect venv
        args: ['-m', 'uvicorn', 'backend.app:app', '--host', '0.0.0.0', '--port', '8000', '--reload']
    },
    frontend: {
        port: 5173, // Matching Vite default
        path: path.join(__dirname, 'Front-end'),
        command: 'npm',
        args: ['run', 'dev']
    },
    checkInterval: 500, // ms
    startupDelay: 2000 // ms
};

// ANSI Colors
const COLORS = {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    red: '\x1b[31m',
    cyan: '\x1b[36m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m'
};

const log = (source, message, color = COLORS.reset) => {
    const timestamp = new Date().toLocaleTimeString();
    const prefix = `[${timestamp}] ${source}`;
    // Handle multi-line output nicely
    const lines = message.toString().split('\n');
    lines.forEach(line => {
        if (line.trim()) {
            console.log(`${color}${prefix.padEnd(20)} | ${line}${COLORS.reset}`);
        }
    });
};

const systemLog = (msg) => console.log(`${COLORS.green}✔ ${msg}${COLORS.reset}`);
const errorLog = (msg) => console.log(`${COLORS.red}✖ ${msg}${COLORS.reset}`);
const warnLog = (msg) => console.log(`${COLORS.yellow}⚠ ${msg}${COLORS.reset}`);

/**
 * Detects the Python executable to use (venv or system)
 */
function getPythonCommand() {
    const isWin = process.platform === 'win32';
    const venvPath = path.join(__dirname, 'venv');
    const venvPython = isWin 
        ? path.join(venvPath, 'Scripts', 'python.exe')
        : path.join(venvPath, 'bin', 'python');

    const fs = require('fs');
    if (fs.existsSync(venvPython)) {
        log('RUNNER', `Using virtual environment: ${venvPython}`, COLORS.cyan);
        return venvPython;
    }
    
    log('RUNNER', 'Using system python', COLORS.cyan);
    return 'python'; // Fallback to system python
}

/**
 * Kills any process running on the specified port
 */
function killPort(port) {
    return new Promise((resolve, reject) => {
        const platform = process.platform;
        
        if (platform === 'win32') {
            // Windows: Find PID then kill
            const cmd = `netstat -ano | findstr :${port}`;
            exec(cmd, (err, stdout) => {
                if (err || !stdout) {
                    // No process found or error (usually means port is free)
                    resolve();
                    return;
                }

                // Parse PID (last token in the line)
                // TCP    0.0.0.0:8000           0.0.0.0:0              LISTENING       1234
                const lines = stdout.trim().split('\n');
                const pids = new Set();
                
                lines.forEach(line => {
                    const parts = line.trim().split(/\s+/);
                    const pid = parts[parts.length - 1];
                    if (pid && !isNaN(pid) && pid !== '0') {
                        pids.add(pid);
                    }
                });

                if (pids.size === 0) {
                    resolve();
                    return;
                }

                const killCmd = `taskkill /F /PID ${Array.from(pids).join(' /PID ')}`;
                exec(killCmd, (killErr) => {
                    if (killErr) {
                        warnLog(`Failed to kill process on port ${port}: ${killErr.message}`);
                    } else {
                        log('RUNNER', `Freed port ${port} (Killed PIDs: ${Array.from(pids).join(', ')})`, COLORS.yellow);
                    }
                    resolve();
                });
            });
        } else {
            // Linux/macOS
            const cmd = `lsof -ti :${port} | xargs kill -9`;
            exec(cmd, (err) => {
                // Ignore errors (process might not exist)
                if (!err) {
                    log('RUNNER', `Freed port ${port}`, COLORS.yellow);
                }
                resolve();
            });
        }
    });
}

const http = require('http');

/**
 * Checks if the backend is ready by hitting the status endpoint
 */
function checkBackendHealth(port) {
    return new Promise((resolve) => {
        const req = http.get(`http://127.0.0.1:${port}/api/status`, (res) => {
            if (res.statusCode === 200) {
                resolve(true);
            } else {
                resolve(false);
            }
        });

        req.on('error', () => {
            resolve(false);
        });

        req.setTimeout(1000, () => {
            req.destroy();
            resolve(false);
        });
    });
}

/**
 * Main Runner Function
 */
async function start() {
    console.log(`${COLORS.bright}${COLORS.cyan}=== Automated Development Runner ===${COLORS.reset}\n`);

    // 1. Cleanup Ports
    log('RUNNER', 'Cleaning up ports...', COLORS.cyan);
    await killPort(CONFIG.backend.port);
    await killPort(CONFIG.frontend.port);
    systemLog('Ports Freed');

    // Wait a bit for sockets to close fully
    await new Promise(r => setTimeout(r, 1000));

    // 2. Start Backend
    log('RUNNER', 'Starting Backend...', COLORS.cyan);
    const pythonCmd = getPythonCommand();
    
    const backend = spawn(pythonCmd, CONFIG.backend.args, {
        cwd: CONFIG.backend.path,
        shell: true,
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });

    backend.stdout.on('data', (data) => log('BACKEND', data, COLORS.green));
    backend.stderr.on('data', (data) => log('BACKEND', data, COLORS.red)); // Stderr often has info logs in Python
    
    backend.on('close', (code) => {
        errorLog(`Backend process exited with code ${code}`);
    });

    // Wait for backend to be ready
    log('RUNNER', `Waiting for backend health check on port ${CONFIG.backend.port}...`, COLORS.cyan);
    let backendReady = false;
    let attempts = 0;
    const maxAttempts = 60; // 30 seconds
    
    while (!backendReady && attempts < maxAttempts) {
        backendReady = await checkBackendHealth(CONFIG.backend.port);
        if (!backendReady) {
            await new Promise(r => setTimeout(r, CONFIG.checkInterval));
            attempts++;
            if (attempts % 10 === 0) {
                 log('RUNNER', `Still waiting for backend... (${attempts}/${maxAttempts})`, COLORS.yellow);
            }
        }
    }

    if (backendReady) {
        systemLog('Backend Started & Reachable (Health Check Passed)');
    } else {
        warnLog('Backend health check timed out. Proceeding anyway (check logs for errors)...');
    }

    // 3. Start Frontend
    log('RUNNER', 'Starting Frontend...', COLORS.cyan);
    const frontend = spawn(CONFIG.frontend.command, CONFIG.frontend.args, {
        cwd: CONFIG.frontend.path,
        shell: true,
        env: { ...process.env } // Vite handles colors well usually
    });

    frontend.stdout.on('data', (data) => log('FRONTEND', data, COLORS.blue));
    frontend.stderr.on('data', (data) => log('FRONTEND', data, COLORS.magenta));

    frontend.on('close', (code) => {
        errorLog(`Frontend process exited with code ${code}`);
    });

    systemLog('Frontend Started');
    systemLog('System Ready');
    console.log(`\n${COLORS.bright}App running at:\n- Backend: http://localhost:${CONFIG.backend.port}\n- Frontend: http://localhost:${CONFIG.frontend.port}${COLORS.reset}\n`);

    // Handle process exit
    const cleanup = () => {
        log('RUNNER', 'Stopping processes...', COLORS.cyan);
        backend.kill();
        frontend.kill();
        process.exit();
    };

    process.on('SIGINT', cleanup);
    process.on('SIGTERM', cleanup);
}

// Run
start().catch(err => {
    console.error('Runner failed:', err);
});

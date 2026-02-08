import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { CameraFeed } from '@/app/components/CameraFeed';
import { BackendStatus } from '@/app/components/BackendStatus';
import { VoiceStatus } from '@/app/components/VoiceStatus';

type Emotion = {
  name: string;
  emoji: string;
  confidence: number;
};

type MicroExpressions = {
  stress_index: number;
  behavioral_stress_index?: number;
  blink_rate: number;
  blink_stress_level?: string;
  blink_stress_score?: number;
  gaze: string;
  flags: string[];
  micro_expressions: string[];
  lip_compression?: boolean;
  lip_compression_score?: number;
  fps?: number;
};

type AnalysisData = {
  emotion: string | null;
  confidence: number;
  faceDetected: boolean;
  micro_expressions: MicroExpressions | null;
  timestamp: string;
};

type Status = 'connected' | 'processing' | 'disconnected';

const emotionMap: Record<string, string> = {
  'Angry': '😠',
  'Disgust': '🤢',
  'Fear': '😨',
  'Happy': '😄',
  'Neutral': '😐',
  'Sad': '😢',
  'Surprise': '😲'
};

export default function App() {
  const [currentEmotion, setCurrentEmotion] = useState<Emotion | null>(null);
  const [microData, setMicroData] = useState<MicroExpressions | null>(null);
  const [status, setStatus] = useState<Status>('disconnected');
  const [cameraActive, setCameraActive] = useState(false);
  const [backendUrl] = useState('http://localhost:8000');
  const ws = useRef<WebSocket | null>(null);
  const lastUpdateRef = useRef<number>(0);
  const lastEmotionChangeRef = useRef<number>(0);
  const pendingEmotionRef = useRef<Emotion | null>(null);
  const [blinkStats, setBlinkStats] = useState<{count: number; perMin: number | null}>({count: 0, perMin: null});
  const [lipState, setLipState] = useState<string>('lip_calm');
  const [lipConfidence, setLipConfidence] = useState<number>(0);
  
  // Stable flags (min 3s visibility)
  const [displayedFlags, setDisplayedFlags] = useState<{text: string; ts: number}[]>([]);

  // Stress History for Graph (Last 60 points for smoother timeline)
  const [stressHistory, setStressHistory] = useState<number[]>(new Array(60).fill(0));

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await fetch(`${backendUrl}/api/status`);
        const data = await response.json();
        if (data.backendConnected) {
          setStatus('connected');
          setCameraActive(data.videoStreamActive);
          connectWebSocket();
        }
      } catch (error) {
        setStatus('disconnected');
        setCameraActive(false);
      }
    };

    checkStatus();
    const statusInterval = setInterval(checkStatus, 5000);

    return () => {
      clearInterval(statusInterval);
      if (ws.current) ws.current.close();
    };
  }, [backendUrl]);

  const connectWebSocket = () => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) return;

    const wsUrl = backendUrl.replace('http', 'ws') + '/api/ws';
    ws.current = new WebSocket(wsUrl);

    ws.current.onopen = () => setStatus('connected');
    
    ws.current.onmessage = (event) => {
      try {
        // Throttle UI updates to ~3 FPS
        const now = Date.now();
        if (now - lastUpdateRef.current < 300) return;
        lastUpdateRef.current = now;
        
        const data: AnalysisData = JSON.parse(event.data);
        
        // Update Emotion
        if (data.faceDetected && data.emotion) {
          setStatus('processing');
          const nextEmotion: Emotion = {
            name: data.emotion,
            emoji: emotionMap[data.emotion] || '😐',
            confidence: Math.round(data.confidence * 100)
          };
          if (!currentEmotion || currentEmotion.name === nextEmotion.name) {
            setCurrentEmotion(nextEmotion);
            lastEmotionChangeRef.current = now;
            pendingEmotionRef.current = null;
          } else {
            // Freeze dominant expression for 2s before switching
            if (now - lastEmotionChangeRef.current >= 2000) {
              setCurrentEmotion(nextEmotion);
              lastEmotionChangeRef.current = now;
              pendingEmotionRef.current = null;
            } else {
              pendingEmotionRef.current = nextEmotion;
            }
          }
        } else {
          setCurrentEmotion(null);
          setStatus('connected');
        }

        // Update Micro-Expressions
        if (data.micro_expressions) {
          setMicroData(data.micro_expressions);
          setStressHistory(prev => {
            const newHist = [...prev.slice(1), data.micro_expressions!.stress_index];
            return newHist;
          });
        } else {
            setMicroData(null);
        }
          
        // Stable flags: keep visible >= 3s, limit to 5 (Run always to handle expiration)
        setDisplayedFlags(prev => {
        const incoming = (data.micro_expressions?.flags || []).slice(0, 5);
        const nowTs = Date.now();
        const map = new Map<string, number>(prev.map(p => [p.text, p.ts]));
        incoming.forEach(f => {
            if (!map.has(f)) map.set(f, nowTs);
        });
        // remove only if absent and visible >= 3000ms
        const next: {text: string; ts: number}[] = [];
        map.forEach((ts, text) => {
            if (incoming.includes(text) || nowTs - ts < 3000) {
            next.push({ text, ts });
            }
        });
        // limit to 5
        next.sort((a, b) => a.ts - b.ts);
        return next.slice(0, 5);
        });
        
        if (typeof (data as any).blink_count === 'number') {
          const perMin = data.micro_expressions && typeof data.micro_expressions.blink_rate === 'number'
            ? Math.round(data.micro_expressions.blink_rate)
            : null;
          setBlinkStats({ count: (data as any).blink_count || 0, perMin });
        }
        
      } catch (e) {
        console.error('Error parsing WS message', e);
      }
    };

    ws.current.onclose = () => setStatus('disconnected');
    ws.current.onerror = () => setStatus('disconnected');
  };

  useEffect(() => {
    const i = setInterval(async () => {
      try {
        const r = await fetch(`${backendUrl}/api/get_blink_stats`);
        const d = await r.json();
        setBlinkStats({ count: d.blink_count || 0, perMin: d.blink_per_min || null });
        setMicroData(prev => prev ? { ...prev, blink_rate: d.blink_per_min ? Math.round(d.blink_per_min) : 0 } : prev);
      } catch {}
    }, 2000);
    return () => clearInterval(i);
  }, [backendUrl]);

  useEffect(() => {
    const i = setInterval(async () => {
      try {
        const r = await fetch(`${backendUrl}/api/predict_frame`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
        const d = await r.json();
        setLipState(d.lip_state || 'lip_calm');
        setLipConfidence(Math.round((d.confidence || 0) * 100));
      } catch {}
    }, 2000);
    return () => clearInterval(i);
  }, [backendUrl]);

  // Helper for stress color
  const getStressColor = (score: number) => {
    if (score < 30) return 'text-green-500';
    if (score < 60) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getStressBg = (score: number) => {
    if (score < 30) return 'bg-green-500';
    if (score < 60) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-red-900/30 overflow-hidden flex flex-col">
      {/* 1. Header & System Status */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
        {/* Disclaimer */}
        <div className="bg-red-950/50 border-b border-red-900/30 px-4 py-1 text-[10px] text-red-300 font-mono text-center uppercase tracking-widest">
          ⚠ Restricted Access: Interrogation Support System • AI Behavioral Insights Only • Not Admissible as Truth Detection
        </div>
        
        <div className="max-w-[1920px] mx-auto px-6 py-3 flex justify-between items-center">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
               <div className={`h-3 w-3 rounded-full animate-pulse shadow-[0_0_10px_rgba(239,68,68,0.5)] ${status === 'processing' ? 'bg-green-500' : status === 'connected' ? 'bg-yellow-500' : 'bg-red-500'}`} />
               <h1 className="text-lg font-bold tracking-tight text-slate-100">
                 SENTINEL <span className="text-slate-600 font-light mx-2">|</span> <span className="text-slate-400 font-normal text-sm">Behavioral Analysis Unit</span>
               </h1>
            </div>
          </div>

          <div className="flex items-center gap-6 text-xs font-mono text-slate-500">
            <div className="flex items-center gap-2">
              <span className={`h-1.5 w-1.5 rounded-full ${status !== 'disconnected' ? 'bg-green-500' : 'bg-red-500'}`} />
              BACKEND
            </div>
            <div className="flex items-center gap-2">
              <span className={`h-1.5 w-1.5 rounded-full ${cameraActive ? 'bg-green-500' : 'bg-red-500'}`} />
              CAMERA
            </div>
            <div className="flex items-center gap-2">
              <span className={`h-1.5 w-1.5 rounded-full ${status === 'processing' ? 'bg-green-500' : 'bg-slate-700'}`} />
              ANALYSIS
            </div>
            <div className="flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-slate-700" />
              FPS: {microData?.fps ?? '--'}
            </div>
            <BackendStatus status={status} />
          </div>
        </div>
      </header>

      {/* 2. Main Dashboard Grid */}
      <main className="flex-1 p-4 lg:p-6 overflow-hidden">
        <div className="h-full grid grid-cols-1 lg:grid-cols-10 gap-6 max-w-[1920px] mx-auto">
          
          {/* LEFT COLUMN: Camera Feed (60% width -> col-span-6) */}
          <div className="lg:col-span-6 flex flex-col gap-4 h-full">
            <div className="relative flex-1 rounded-2xl overflow-hidden border border-slate-800 bg-black shadow-2xl group">
              <CameraFeed isActive={true} videoSource={`${backendUrl}/api/video-stream`} />
              
              {/* Camera Overlays */}
              <div className="absolute inset-0 pointer-events-none p-6 flex flex-col justify-between">
                <div className="flex justify-between items-start">
                   <div className="px-3 py-1.5 rounded-md bg-black/70 backdrop-blur border border-green-900/50 flex items-center gap-3">
                      <div className="h-2 w-2 bg-red-500 rounded-full animate-pulse" />
                      <span className="text-xs font-mono text-green-400 tracking-wider">
                        LIVE FEED: CAM-01 {microData?.fps ? `• ${microData.fps} FPS` : ''}
                      </span>
                   </div>
                   {microData && (
                     <div className="px-3 py-1.5 rounded-md bg-black/70 backdrop-blur border border-slate-700">
                        <span className="text-xs font-mono text-slate-400 mr-2">GAZE DIRECTION</span>
                        <span className="text-sm font-bold text-white">{microData.gaze.toUpperCase()}</span>
                     </div>
                   )}
                </div>

                {/* Face Bounding Box Overlay (Visual only, actual box is drawn by backend) */}
                <div className="absolute inset-0 border-[1px] border-slate-800/20 m-12 rounded-lg" />
                
                {/* Bottom Stats Overlay */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-black/80 backdrop-blur p-3 rounded-lg border border-slate-800/50">
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">Expression Level</div>
                    <div className="text-xl font-mono text-blue-400">{currentEmotion?.confidence || 0}%</div>
                  </div>
                  <div className="bg-black/80 backdrop-blur p-3 rounded-lg border border-slate-800/50">
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">Face Detected</div>
                    <div className={`text-xl font-mono ${currentEmotion ? 'text-green-400' : 'text-red-400'}`}>
                      {currentEmotion ? 'YES' : 'NO'}
                    </div>
                  </div>
                  <div className="bg-black/80 backdrop-blur p-3 rounded-lg border border-slate-800/50">
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">Blink Rate</div>
                    <div className="text-xl font-mono text-green-400">{microData?.blink_rate || 0} bpm</div>
                    <div className="mt-1 text-[10px] font-mono text-slate-400">
                      {microData?.blink_stress_level ? `Status: ${microData.blink_stress_level}` : ''}
                    </div>
                    <div className="mt-1 text-[10px] font-mono text-slate-400">
                      {`Count: ${blinkStats.count}`}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* CENTER COLUMN: Analysis & Stress (20% -> col-span-2) */}
          <div className="lg:col-span-2 flex flex-col gap-4 h-full overflow-y-auto pr-2">
            
            {/* Voice Status Component */}
            <VoiceStatus />

            {/* Dominant Expression Card */}
            <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-6 flex flex-col items-center justify-center text-center relative overflow-hidden">
               <div className="absolute inset-0 bg-gradient-to-b from-slate-800/20 to-transparent pointer-events-none" />
               <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 z-10">Dominant Expression</h3>
               
               <div className="relative z-10 mb-2">
                 <span className="text-8xl filter drop-shadow-[0_0_15px_rgba(255,255,255,0.1)] transition-transform duration-300 hover:scale-110 block">
                   {currentEmotion?.emoji || '⌛'}
                 </span>
               </div>
               
               <div className="z-10 mt-2">
                 <div className="text-2xl font-bold text-white tracking-tight">{currentEmotion?.name || 'Waiting...'}</div>
                 <div className="text-xs text-slate-400 mt-1">Real-time Classification</div>
                 {/* Expression Level Bar */}
                 <div className="mt-3 w-full">
                   <div className="flex items-center justify-between mb-1">
                     <span className="text-[10px] text-slate-500 uppercase tracking-widest">Expression Level</span>
                     <span className="text-[10px] text-slate-400 font-mono">{currentEmotion?.confidence || 0}%</span>
                   </div>
                   <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                     <motion.div 
                       initial={{ width: 0 }}
                       animate={{ width: `${currentEmotion?.confidence || 0}%` }}
                       className="h-full bg-blue-500"
                       transition={{ duration: 0.3 }}
                     />
                   </div>
                 </div>
               </div>
            </div>

            {/* Stress Index */}
            <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-6">
              <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Behavioral Stress Index</h3>
              
              <div className="flex items-end gap-2 mb-2">
                <span className={`text-5xl font-black font-mono tracking-tighter ${getStressColor(microData?.behavioral_stress_index || microData?.stress_index || 0)}`}>
                  {Math.round(microData?.behavioral_stress_index || microData?.stress_index || 0)}
                </span>
                <span className="text-sm text-slate-500 font-mono mb-2">/100</span>
              </div>
              
              {/* Horizontal Bar */}
              <div className="h-3 bg-slate-800 rounded-full overflow-hidden mb-2">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: `${microData?.behavioral_stress_index || microData?.stress_index || 0}%` }}
                  className={`h-full ${getStressBg(microData?.behavioral_stress_index || microData?.stress_index || 0)}`}
                  transition={{ type: "spring", stiffness: 50 }}
                />
              </div>
              <div className="flex justify-between text-[10px] font-mono text-slate-500">
                <span>NORMAL</span>
                <span>WARN</span>
                <span>CRITICAL</span>
              </div>
            </div>

            {/* Expanded Timeline */}
            <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-4 h-64 flex flex-col shrink-0">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Stress Timeline (60s)</h3>
                <span className="px-2 py-0.5 bg-slate-800 text-[10px] rounded text-slate-400">Fixed Scale: 0-100</span>
              </div>
              <div className="flex-1 flex items-end gap-[1px] border-b border-slate-700/50 pb-1 relative overflow-hidden">
                {/* Grid lines */}
                <div className="absolute inset-0 flex flex-col justify-between pointer-events-none opacity-20">
                   <div className="w-full border-t border-slate-400 border-dashed" />
                   <div className="w-full border-t border-slate-400 border-dashed" />
                   <div className="w-full border-t border-slate-400 border-dashed" />
                </div>
                
                {stressHistory.map((val, i) => (
                  <div 
                    key={i} 
                    className="flex-1 bg-slate-700/30 rounded-t-[1px] hover:bg-white/20 transition-colors relative group"
                    style={{ height: `${Math.min(100, Math.max(0, val))}%` }}
                  >
                     <div className={`absolute bottom-0 left-0 right-0 top-0 opacity-60 ${getStressBg(val)}`} />
                     {/* Tooltip */}
                     <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-black text-xs rounded opacity-0 group-hover:opacity-100 whitespace-nowrap z-50 pointer-events-none">
                       Stress: {Math.round(val)}
                     </div>
                  </div>
                ))}
              </div>
              <div className="flex justify-between mt-2 text-[10px] text-slate-600 font-mono">
                <span>-60s</span>
                <span>Now</span>
              </div>
            </div>

          </div>

          {/* RIGHT COLUMN: Flags & Biometrics (20% -> col-span-2) */}
          <div className="lg:col-span-2 flex flex-col gap-4 h-full overflow-y-auto pr-2">
            
            {/* Biometrics Panel */}
            <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-5">
              <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Biometrics</h3>
              <div className="space-y-4">
                {/* Blink Rate */}
                <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <span className="text-lg">👁</span>
                    <div>
                      <div className="text-[10px] text-slate-400 uppercase">Blink Rate</div>
                      <div className="text-sm font-bold text-slate-200">
                        {microData?.blink_rate || 0} <span className="text-[10px] font-normal text-slate-500">bpm</span>
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5">
                        {microData?.blink_stress_level || 'Normal'}
                      </div>
                      {/* Stress bar */}
                      <div className="mt-1 w-40 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.min(100, microData?.blink_stress_score || 0)}%` }}
                          className={`h-full ${getStressBg(microData?.blink_stress_score || 0)}`}
                          transition={{ type: 'spring', stiffness: 60 }}
                        />
                      </div>
                    </div>
                  </div>
                  <div className={`h-2 w-2 rounded-full ${
                    (microData?.blink_stress_level || 'Normal') === 'High Stress' ? 'bg-red-500' :
                    (microData?.blink_stress_level || 'Normal') === 'Stress' ? 'bg-yellow-500' : 'bg-green-500'
                  }`} />
                </div>

                {/* Eye Contact */}
                <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                   <div className="flex items-center gap-3">
                    <span className="text-lg">🎯</span>
                    <div>
                      <div className="text-[10px] text-slate-400 uppercase">Eye Contact</div>
                      <div className="text-sm font-bold text-slate-200">
                        {microData && microData.gaze === "Center" ? "Stable" : "Evasive"}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Head Pose */}
                <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                   <div className="flex items-center gap-3">
                    <span className="text-lg">👤</span>
                    <div>
                      <div className="text-[10px] text-slate-400 uppercase">Head Pose</div>
                      <div className="text-sm font-bold text-slate-200">
                         {microData?.flags.some(f => f.includes("Head")) ? "Active" : "Neutral"}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Lip Compression */}
                <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <span className="text-lg">👄</span>
                    <div>
                      <div className="text-[10px] text-slate-400 uppercase">Lip Compression</div>
                      <div className="text-sm font-bold text-slate-200">
                        {(microData?.lip_compression_score || 0) > 60 ? "Detected" : "Calm"}
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5">
                        {`Lip State: ${lipState.replaceAll('_',' ')}`}
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5">
                        {`Confidence: ${lipConfidence}%`}
                      </div>
                      <div className="mt-1 w-40 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.min(100, microData?.lip_compression_score || 0)}%` }}
                          className={`h-full ${
                            (microData?.lip_compression_score || 0) > 90 ? 'bg-red-500' :
                            (microData?.lip_compression_score || 0) > 75 ? 'bg-orange-500' :
                            (microData?.lip_compression_score || 0) > 60 ? 'bg-yellow-500' : 'bg-green-500'
                          }`}
                          transition={{ type: 'spring', stiffness: 60 }}
                        />
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5">
                        Score: {Math.round(microData?.lip_compression_score || 0)}
                      </div>
                    </div>
                  </div>
                  <div className={`h-2 w-2 rounded-full ${
                    (microData?.lip_compression_score || 0) > 90 ? 'bg-red-500' :
                    (microData?.lip_compression_score || 0) > 75 ? 'bg-orange-500' :
                    (microData?.lip_compression_score || 0) > 60 ? 'bg-yellow-500' : 'bg-green-500'
                  }`} />
                </div>
              </div>
            </div>

            {/* Micro-Expression Flags */}
            <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-5 flex-1 flex flex-col">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Flags</h3>
                <span className="px-2 py-0.5 rounded-full bg-slate-800 text-[10px] text-slate-400">
                  {displayedFlags.length || 0}
                </span>
              </div>
              
              <div className="space-y-2 pr-1 custom-scrollbar min-h-0 h-[250px] overflow-y-auto">
                 <AnimatePresence>
                   {displayedFlags && displayedFlags.length > 0 ? (
                     displayedFlags.map((item, idx) => {
                       const flag = item.text;
                       // Determine severity
                       let severityColor = 'border-yellow-900/30 bg-yellow-950/30 text-yellow-200';
                       let icon = '⚠';
                       if (flag.includes("Stress") || flag.includes("Guilt")) {
                         severityColor = 'border-red-900/30 bg-red-950/30 text-red-200';
                         icon = '🚨';
                       } else if (flag.includes("Curiosity")) {
                         severityColor = 'border-blue-900/30 bg-blue-950/30 text-blue-200';
                         icon = 'ℹ';
                       }

                       return (
                         <motion.div 
                           key={`${flag}-${idx}`}
                           initial={{ opacity: 0, x: 20 }}
                           animate={{ opacity: 1, x: 0 }}
                           exit={{ opacity: 0, x: -20 }}
                           className={`p-3 rounded-lg border text-xs font-medium flex items-start gap-3 ${severityColor}`}
                         >
                           <span className="mt-0.5">{icon}</span>
                           <div className="flex-1 leading-relaxed">
                             {flag}
                           </div>
                         </motion.div>
                       );
                     })
                   ) : (
                     <div className="h-full flex flex-col items-center justify-center text-slate-600 gap-2 opacity-50">
                       <span className="text-2xl">🛡</span>
                       <span className="text-xs">No anomalies detected</span>
                     </div>
                   )}
                 </AnimatePresence>
              </div>
            </div>

          </div>

        </div>
      </main>
    </div>
  );
}

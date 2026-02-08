
import { useState, useEffect } from 'react';
import { motion } from 'motion/react';

type VoiceData = {
  model_loaded: boolean;
  language: string;
  arousal: number;
  dominance: number;
  valence: number;
  stress_level: string;
  stress_score: number;
};

export function VoiceStatus() {
  const [data, setData] = useState<VoiceData | null>(null);
  const [error, setError] = useState<boolean>(false);

  useEffect(() => {
    const fetchVoiceStatus = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/voice/status');
        if (response.ok) {
          const json = await response.json();
          setData(json);
          setError(false);
        } else {
          setError(true);
        }
      } catch (e) {
        setError(true);
      }
    };

    const interval = setInterval(fetchVoiceStatus, 500); // Poll every 500ms
    fetchVoiceStatus();

    return () => clearInterval(interval);
  }, []);

  if (error || !data) {
    return (
      <div className="bg-gray-800/50 backdrop-blur-md border border-gray-700/50 rounded-xl p-4 shadow-xl">
        <h3 className="text-gray-400 text-sm font-medium mb-2 uppercase tracking-wider">Voice Stress</h3>
        <div className="flex items-center space-x-2 text-yellow-500">
          <span className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></span>
          <span className="text-sm">Connecting / Loading Model...</span>
        </div>
      </div>
    );
  }

  const isStress = data.stress_level === 'HIGH';
  const stressColor = isStress ? 'text-red-400' : 'text-emerald-400';
  const stressBg = isStress ? 'bg-red-500/20' : 'bg-emerald-500/20';
  const stressBorder = isStress ? 'border-red-500/30' : 'border-emerald-500/30';

  return (
    <div className={`backdrop-blur-md border rounded-xl p-4 shadow-xl transition-colors duration-500 ${isStress ? 'bg-red-900/10 border-red-500/30' : 'bg-gray-800/50 border-gray-700/50'}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-gray-400 text-sm font-medium uppercase tracking-wider">Voice Analysis (Real-Time)</h3>
        {data.model_loaded && (
          <span className="px-2 py-0.5 rounded text-xs bg-blue-500/20 text-blue-300 border border-blue-500/30">
            Model Active
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Stress Indicator */}
        <div className={`col-span-2 p-3 rounded-lg border ${stressBorder} ${stressBg} flex items-center justify-between`}>
          <div>
            <div className="text-xs text-gray-400 uppercase tracking-wider mb-1">Current State</div>
            <div className={`text-xl font-bold ${stressColor} tracking-wide`}>
              {data.stress_level}
            </div>
          </div>
          <div className="text-right">
             <div className="text-xs text-gray-400 uppercase tracking-wider mb-1">Stress Score</div>
             <div className="text-2xl font-mono font-bold text-white">
               {(data.stress_score * 100).toFixed(0)}<span className="text-sm text-gray-500">%</span>
             </div>
          </div>
        </div>

        {/* Metrics */}
        <div className="space-y-3 col-span-2 bg-black/20 p-3 rounded-lg">
           <div className="flex justify-between items-center text-sm">
             <span className="text-gray-400">Arousal (Energy)</span>
             <span className="font-mono text-white">{data.arousal.toFixed(2)}</span>
           </div>
           <div className="w-full bg-gray-700 h-1.5 rounded-full overflow-hidden">
             <motion.div 
               className="h-full bg-orange-400"
               initial={{ width: 0 }}
               animate={{ width: `${Math.min(100, Math.max(0, data.arousal * 100))}%` }}
               transition={{ type: "spring", stiffness: 100 }}
             />
           </div>

           <div className="flex justify-between items-center text-sm mt-2">
             <span className="text-gray-400">Dominance (Control)</span>
             <span className="font-mono text-white">{data.dominance.toFixed(2)}</span>
           </div>
           <div className="w-full bg-gray-700 h-1.5 rounded-full overflow-hidden">
             <motion.div 
               className="h-full bg-purple-400"
               initial={{ width: 0 }}
               animate={{ width: `${Math.min(100, Math.max(0, data.dominance * 100))}%` }}
               transition={{ type: "spring", stiffness: 100 }}
             />
           </div>

           <div className="flex justify-between items-center text-sm mt-2">
             <span className="text-gray-400">Valence (Positivity)</span>
             <span className="font-mono text-white">{data.valence.toFixed(2)}</span>
           </div>
           <div className="w-full bg-gray-700 h-1.5 rounded-full overflow-hidden">
             <motion.div 
               className="h-full bg-blue-400"
               initial={{ width: 0 }}
               animate={{ width: `${Math.min(100, Math.max(0, data.valence * 100))}%` }}
               transition={{ type: "spring", stiffness: 100 }}
             />
           </div>
        </div>
      </div>
      
      <div className="mt-3 text-[10px] text-gray-500 flex justify-between">
         <span>Model: wav2vec2-emotion</span>
         <span>Lang: {data.language.toUpperCase()}</span>
      </div>
    </div>
  );
}

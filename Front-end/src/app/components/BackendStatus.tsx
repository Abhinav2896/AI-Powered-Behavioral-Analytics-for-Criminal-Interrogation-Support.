import { motion } from 'motion/react';

type Status = 'connected' | 'processing' | 'disconnected';

interface BackendStatusProps {
  status: Status;
}

const statusConfig = {
  connected: {
    color: '#22C55E',
    label: 'ONLINE',
    animate: false,
  },
  processing: {
    color: '#FACC15',
    label: 'PROCESSING',
    animate: true,
  },
  disconnected: {
    color: '#EF4444',
    label: 'OFFLINE',
    animate: false,
  },
};

export function BackendStatus({ status }: BackendStatusProps) {
  const config = statusConfig[status];

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/50 border border-slate-700/50">
      <div className="relative flex items-center justify-center w-2 h-2">
        <motion.div
          className="w-2 h-2 rounded-full"
          style={{ background: config.color }}
          animate={config.animate ? { scale: [1, 1.2, 1], opacity: [1, 0.8, 1] } : {}}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
        {config.animate && (
           <motion.div
             className="absolute w-full h-full rounded-full"
             style={{ background: config.color }}
             animate={{ scale: [1, 2], opacity: [0.5, 0] }}
             transition={{ duration: 1.5, repeat: Infinity }}
           />
        )}
      </div>
      <span className="text-[10px] font-bold tracking-wider text-slate-300" style={{ color: config.color }}>
        {config.label}
      </span>
    </div>
  );
}

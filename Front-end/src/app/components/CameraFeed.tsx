import { motion } from 'motion/react';
import { useState, useEffect } from 'react';

interface CameraFeedProps {
  isActive: boolean;
  videoSource?: string;
}

export function CameraFeed({ isActive, videoSource }: CameraFeedProps) {
  const [imgError, setImgError] = useState(false);
  const [retryKey, setRetryKey] = useState(0);

  // Retry loading image every 3 seconds if error occurs
  useEffect(() => {
    if (imgError && isActive) {
      const timer = setTimeout(() => {
        setImgError(false);
        setRetryKey(prev => prev + 1);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [imgError, isActive]);

  // Add timestamp to video source to prevent caching/stuck connections
  const videoSrc = videoSource ? `${videoSource}?t=${retryKey}` : '';

  return (
    <div className="w-full h-full relative bg-black flex items-center justify-center overflow-hidden">
        {/* Camera Feed */}
        {isActive && videoSource && !imgError ? (
          <img 
            key={retryKey}
            src={videoSrc} 
            alt="Live Emotion Detection Feed" 
            className="w-full h-full object-cover"
            onError={() => {
                console.error("Video feed error, retrying...");
                setImgError(true);
            }}
          />
        ) : (
          <div className="flex flex-col items-center gap-4 text-slate-700">
             <div className="text-6xl animate-pulse">📷</div>
             <p className="text-sm font-mono">{imgError ? "SIGNAL LOST" : "CONNECTING..."}</p>
          </div>
        )}

        {/* Scanline Effect */}
        {isActive && !imgError && (
          <div className="absolute inset-0 pointer-events-none opacity-10 bg-[linear-gradient(transparent_50%,rgba(0,0,0,0.5)_50%)] bg-[length:100%_4px]" />
        )}
        
        {/* Animated Scan Bar */}
        {isActive && !imgError && (
           <motion.div 
             className="absolute top-0 left-0 right-0 h-1 bg-green-500/30 shadow-[0_0_20px_rgba(34,197,94,0.5)]"
             animate={{ top: ['0%', '100%', '0%'] }}
             transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
           />
        )}
    </div>
  );
}

import { motion, useMotionValue, useTransform, animate } from 'motion/react';
import { useEffect, useState } from 'react';

interface ConfidenceScoreProps {
  confidence: number; // 0-100
}

export function ConfidenceScore({ confidence }: ConfidenceScoreProps) {
  const [displayValue, setDisplayValue] = useState(0);
  const motionValue = useMotionValue(0);

  useEffect(() => {
    const controls = animate(motionValue, confidence, {
      duration: 0.8,
      ease: 'easeOut',
      onUpdate: (latest) => {
        setDisplayValue(Math.round(latest));
      },
    });

    return controls.stop;
  }, [confidence, motionValue]);

  // Calculate color based on confidence level
  const getColor = (value: number) => {
    if (value < 40) return '#EF4444'; // Red
    if (value < 70) return '#FACC15'; // Yellow
    return '#22C55E'; // Green
  };

  const getGradient = (value: number) => {
    if (value < 40) {
      return 'linear-gradient(90deg, #EF4444 0%, #F87171 100%)';
    }
    if (value < 70) {
      return 'linear-gradient(90deg, #FACC15 0%, #FDE047 100%)';
    }
    return 'linear-gradient(90deg, #22C55E 0%, #4ADE80 100%)';
  };

  return (
    <div
      className="p-6 rounded-2xl"
      style={{
        background: 'rgba(255, 255, 255, 0.05)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      <h3 className="text-[#E5E7EB] text-sm font-semibold mb-4">
        Confidence Level
      </h3>

      <div className="space-y-6">
        {/* Circular Progress */}
        <div className="flex items-center justify-center">
          <div className="relative w-40 h-40">
            {/* Background Circle */}
            <svg className="w-full h-full -rotate-90">
              <circle
                cx="80"
                cy="80"
                r="70"
                stroke="rgba(255, 255, 255, 0.1)"
                strokeWidth="12"
                fill="none"
              />
              {/* Progress Circle */}
              <motion.circle
                cx="80"
                cy="80"
                r="70"
                stroke={getColor(displayValue)}
                strokeWidth="12"
                fill="none"
                strokeLinecap="round"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: displayValue / 100 }}
                transition={{
                  duration: 0.8,
                  ease: 'easeOut',
                }}
                style={{
                  filter: `drop-shadow(0 0 8px ${getColor(displayValue)})`,
                }}
              />
            </svg>
            {/* Center Text */}
            <div className="absolute inset-0 flex items-center justify-center">
              <motion.div
                key={displayValue}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="text-center"
              >
                <div className="text-4xl font-bold text-white">
                  {displayValue}%
                </div>
              </motion.div>
            </div>
          </div>
        </div>

        {/* Horizontal Progress Bar */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm text-[#E5E7EB]">
            <span>Low</span>
            <span>High</span>
          </div>
          <div className="w-full h-3 bg-white/10 rounded-full overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              initial={{ width: '0%' }}
              animate={{ width: `${displayValue}%` }}
              transition={{
                duration: 0.8,
                ease: 'easeOut',
              }}
              style={{
                background: getGradient(displayValue),
                boxShadow: `0 0 12px ${getColor(displayValue)}`,
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

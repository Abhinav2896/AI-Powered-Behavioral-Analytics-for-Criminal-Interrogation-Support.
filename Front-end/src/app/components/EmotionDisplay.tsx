import { motion, AnimatePresence } from 'motion/react';

interface EmotionDisplayProps {
  emotion: string;
  emoji: string;
}

export function EmotionDisplay({ emotion, emoji }: EmotionDisplayProps) {
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
        Detected Expression
      </h3>

      <AnimatePresence mode="wait">
        <motion.div
          key={emotion}
          initial={{ opacity: 0, y: 20, scale: 0.8 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -20, scale: 0.8 }}
          transition={{
            duration: 0.5,
            ease: [0.4, 0, 0.2, 1],
          }}
          className="flex flex-col items-center gap-4"
        >
          <motion.div
            animate={{
              scale: [1, 1.05, 1],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
            className="text-7xl"
          >
            {emoji}
          </motion.div>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-white text-2xl font-semibold"
          >
            {emotion}
          </motion.p>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

import { ReactNode } from 'react';
import { motion } from 'framer-motion';

export default function Card({ children }: { children: ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl p-5 shadow-lg shadow-black/40"
    >
      {children}
    </motion.div>
  );
}

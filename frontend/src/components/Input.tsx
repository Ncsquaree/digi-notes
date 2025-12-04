interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
}

export default function Input({ label, className = '', ...rest }: InputProps) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-gray-300">{label}</span>
      <input
        {...rest}
        className={`bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-sm outline-none focus:border-violet-400 ${className}`}
      />
    </label>
  );
}

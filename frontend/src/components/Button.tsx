interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
}

export default function Button({ children, className = '', ...rest }: ButtonProps) {
  return (
    <button
      {...rest}
      className={`px-4 py-2.5 rounded-xl text-sm font-medium bg-white text-black hover:bg-gray-100 transition shadow-md shadow-white/10 ${className}`}
    >
      {children}
    </button>
  );
}

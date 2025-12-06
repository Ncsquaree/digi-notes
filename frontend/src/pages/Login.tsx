import { FormEvent, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import Button from '../components/Button';
import Input from '../components/Input';
import { parseApiError } from '../utils/errors';

export default function Login() {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<Array<{field: string; message: string}>>([]);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await login(email, password);
      navigate('/dashboard');
    } catch (err: any) {
      const parsed = parseApiError(err);
      setError(parsed.message || err?.message || 'Failed to login');
      setValidationErrors(parsed.validationErrors || []);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-black via-[#050509] to-violet-950">
      <div className="w-full max-w-md rounded-3xl border border-white/10 bg-white/5 backdrop-blur-2xl p-8 shadow-2xl shadow-violet-900/50">
        <div className="flex items-center gap-2 mb-6">
          <div className="h-9 w-9 rounded-2xl bg-violet-500 flex items-center justify-center text-xl font-bold">N</div>
          <div>
            <div className="font-semibold">Notexa</div>
            <div className="text-xs text-gray-400">AI Study Workspace</div>
          </div>
        </div>
        <h1 className="text-2xl font-semibold mb-2">Welcome back</h1>
        <p className="text-sm text-gray-400 mb-6">Sign in to continue building your smart notes.</p>
        {error && (
          <div className="mb-3 text-xs text-red-400 bg-red-950/40 border border-red-500/40 rounded-xl px-3 py-2">
            <div className="font-semibold mb-1">{error}</div>
            {validationErrors.length > 0 && (
              <ul className="list-none space-y-0.5 mt-2 text-red-300">
                {validationErrors.map((err, idx) => (
                  <li key={idx}>• {err.message}</li>
                ))}
              </ul>
            )}
          </div>
        )}
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <Input
            label="Email"
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
          />
          <Input
            label="Password"
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
          />
          <Button type="submit" disabled={loading} className="mt-2">
            {loading ? 'Signing in...' : 'Sign in'}
          </Button>
        </form>
        <p className="mt-6 text-xs text-gray-400">
          New here?{' '}
          <Link to="/signup" className="text-gray-100 underline">
            Create an account
          </Link>
        </p>
      </div>
    </div>
  );
}

import { FormEvent, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import Button from '../components/Button';
import Input from '../components/Input';

export default function Signup() {
  const navigate = useNavigate();
  const { signup } = useAuth();
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await signup(email, password, name);
      navigate('/dashboard');
    } catch (err: any) {
      const msg = err?.response?.data?.message || err?.message || 'Failed to sign up';
      setError(msg);
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
            <div className="text-xs text-gray-400">Create your study OS</div>
          </div>
        </div>
        <h1 className="text-2xl font-semibold mb-2">Create an account</h1>
        <p className="text-sm text-gray-400 mb-6">Organise subjects, chapters and smart notes in one place.</p>
        {error && <div className="mb-3 text-xs text-red-400 bg-red-950/40 border border-red-500/40 rounded-xl px-3 py-2">{error}</div>}
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <Input
            label="Name"
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Ram"
          />
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
            {loading ? 'Creating...' : 'Create account'}
          </Button>
        </form>
        <p className="mt-6 text-xs text-gray-400">
          Already have an account?{' '}
          <Link to="/" className="text-gray-100 underline">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}

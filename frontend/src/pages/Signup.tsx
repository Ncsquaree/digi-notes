import { FormEvent, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import Button from '../components/Button';
import Input from '../components/Input';
import { parseApiError } from '../utils/errors';

export default function Signup() {
  const navigate = useNavigate();
  const { signup } = useAuth();
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<Array<{field: string; message: string}>>([]);
  const [loading, setLoading] = useState(false);

  // Password real-time checks
  const passwordChecks = useMemo(() => ({
    minLength: password.length >= 8,
    hasUpper: /[A-Z]/.test(password),
    hasLower: /[a-z]/.test(password),
    hasNumber: /[0-9]/.test(password),
    hasSpecial: /[!@#$%^&*()_+\-=`~\[\]{};:'"\\|,<.>\/?]/.test(password),
  }), [password]);

  const passwordStrength = useMemo(() => {
    const checks = Object.values(passwordChecks);
    const passed = checks.filter(Boolean).length;
    if (passed <= 2) return { label: 'Weak', color: 'bg-red-500', width: '33%' };
    if (passed <= 4) return { label: 'Medium', color: 'bg-yellow-500', width: '66%' };
    return { label: 'Strong', color: 'bg-green-500', width: '100%' };
  }, [passwordChecks]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setValidationErrors([]);
    setLoading(true);
    try {
      await signup(email, password, name);
      navigate('/dashboard');
    } catch (err: any) {
      const parsed = parseApiError(err);
      setError(parsed.message || err?.message || 'Failed to sign up');
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
            <div className="text-xs text-gray-400">Create your study OS</div>
          </div>
        </div>
        <h1 className="text-2xl font-semibold mb-2">Create an account</h1>
        <p className="text-sm text-gray-400 mb-6">Organise subjects, chapters and smart notes in one place.</p>
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
          <div className="flex flex-col gap-1">
            <Input
              label="Password"
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
            />

            {password && (
              <div className="mt-1">
                <div className="flex items-center gap-2 mb-1">
                  <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className={`${passwordStrength.color} h-full transition-all duration-300`}
                      style={{ width: passwordStrength.width }}
                    />
                  </div>
                  <span className="text-xs text-gray-400">{passwordStrength.label}</span>
                </div>

                <div className="text-xs space-y-0.5 text-gray-400">
                  <div className={passwordChecks.minLength ? 'text-green-400' : ''}>
                    {passwordChecks.minLength ? '✓' : '○'} At least 8 characters
                  </div>
                  <div className={passwordChecks.hasUpper ? 'text-green-400' : ''}>
                    {passwordChecks.hasUpper ? '✓' : '○'} Uppercase letter
                  </div>
                  <div className={passwordChecks.hasLower ? 'text-green-400' : ''}>
                    {passwordChecks.hasLower ? '✓' : '○'} Lowercase letter
                  </div>
                  <div className={passwordChecks.hasNumber ? 'text-green-400' : ''}>
                    {passwordChecks.hasNumber ? '✓' : '○'} Number
                  </div>
                  <div className={passwordChecks.hasSpecial ? 'text-green-400' : ''}>
                    {passwordChecks.hasSpecial ? '✓' : '○'} Special character (optional)
                  </div>
                </div>
              </div>
            )}
          </div>
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

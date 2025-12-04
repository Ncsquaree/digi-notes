import { useEffect, useState } from 'react';
import Layout from '../components/Layout';
import Card from '../components/Card';
import { Link } from 'react-router-dom';
import apiClient from '../api/client';

export default function Dashboard() {
  const [stats, setStats] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetch = async () => {
      setError(null);
      setLoading(true);
      try {
        const resp = await apiClient.get('/dashboard');
        setStats(resp.data?.stats ?? resp.data ?? null);
      } catch (e: any) {
        setError(e?.response?.data?.message || e?.message || 'Failed to load dashboard');
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, []);

  return (
    <Layout>
      <div className="mb-6">
        <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Dashboard</div>
        <div className="text-xl font-semibold mt-1">Your workspace overview</div>
      </div>
      {loading && <div className="p-4">Loading dashboard...</div>}
      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500 text-red-400 rounded mb-4">{error}</div>
      )}
      {!loading && stats && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <Card>
            <div className="flex flex-col gap-3">
              <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Subjects</div>
              <div className="text-lg font-semibold">{stats.total_subjects ?? 0}</div>
              <p className="text-sm text-gray-400">Total subjects created</p>
              <Link to="/subjects" className="inline-flex text-xs px-3 py-1.5 rounded-full bg-violet-500/90 hover:bg-violet-400 text-white w-fit">
                Open subjects →
              </Link>
            </div>
          </Card>
          <Card>
            <div className="flex flex-col gap-3">
              <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Notes</div>
              <div className="text-lg font-semibold">{stats.total_notes ?? 0}</div>
              <p className="text-sm text-gray-400">Saved notes across subjects</p>
              <Link to="/library" className="inline-flex text-xs px-3 py-1.5 rounded-full bg-white text-black hover:bg-gray-100 w-fit">
                Open library →
              </Link>
            </div>
          </Card>
          <Card>
            <div className="flex flex-col gap-3">
              <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Flashcards</div>
              <div className="text-lg font-semibold">{stats.total_flashcards ?? 0}</div>
              <p className="text-sm text-gray-400">Due for review: {stats.due_flashcards ?? 0}</p>
              <Link to="/flashcards" className="inline-flex text-xs px-3 py-1.5 rounded-full bg-white/10 hover:bg-white/20 text-gray-100 w-fit">
                Open flashcards →
              </Link>
            </div>
          </Card>
        </div>
      )}

      {!loading && stats && (
        <div className="mt-6">
          <div className="text-sm font-semibold mb-2">Recent subjects</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {stats.recent_subjects && stats.recent_subjects.length ? (
              stats.recent_subjects.map((s: any) => (
                <Card key={s.id}>
                  <div>
                    <div className="font-medium">{s.name}</div>
                    <div className="text-xs text-gray-400">{s.description}</div>
                  </div>
                </Card>
              ))
            ) : (
              <div className="text-xs text-gray-500">No recent subjects</div>
            )}
          </div>
        </div>
      )}
    </Layout>
  );
}

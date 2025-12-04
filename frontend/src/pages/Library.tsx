import { useEffect, useState } from 'react';
import Layout from '../components/Layout';
import Card from '../components/Card';
import apiClient from '../api/client';
import axios from 'axios';

export default function Library() {
  const [notes, setNotes] = useState<any[]>([]);
  const [flashcards, setFlashcards] = useState<any[]>([]);
  const [sessions, setSessions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'notes' | 'flashcards' | 'sessions'>('all');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);

  const fetchAll = async () => {
    setError(null);
    setLoading(true);
    try {
      const [nRes, fRes, sRes] = await Promise.all([
        apiClient.get('/notes'),
        apiClient.get('/flashcards'),
        apiClient.get('/study-sessions')
      ]);
      setNotes(nRes.data?.notes ?? nRes.data?.data ?? nRes.data ?? []);
      setFlashcards(fRes.data?.flashcards ?? fRes.data?.data ?? fRes.data ?? []);
      setSessions(sRes.data?.sessions ?? sRes.data?.data ?? sRes.data ?? []);
    } catch (e: any) {
      setError(e?.response?.data?.message || e?.message || 'Failed to load library');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAll();
  }, []);

  const filtered = () => {
    const q = searchQuery.trim().toLowerCase();
    const out: any[] = [];
    if (filterType === 'all' || filterType === 'notes') {
      out.push(...notes.filter(n => !q || (n.title || '').toLowerCase().includes(q)));
    }
    if (filterType === 'all' || filterType === 'flashcards') {
      out.push(...flashcards.filter(f => !q || (f.question || '').toLowerCase().includes(q)));
    }
    if (filterType === 'all' || filterType === 'sessions') {
      out.push(...sessions.filter(s => {
        if (!q) return true;
        // search over real session fields: quality, time_spent_seconds, reviewed_at
        const repr = `${s.quality ?? ''} ${s.time_spent_seconds ?? ''} ${s.reviewed_at ?? ''}`.toLowerCase();
        return repr.includes(q);
      }));
    }
    return out;
  };

  const handleUpload = async () => {
    if (!uploadFile) return;
    setUploading(true);
    setError(null);
    try {
      const presignResp = await apiClient.post('/s3/presign-upload', { filename: uploadFile.name, contentType: uploadFile.type });
      const { uploadUrl, key } = presignResp.data ?? presignResp.data?.data ?? {};
      if (!uploadUrl || !key) throw new Error('Invalid presign response');

      await axios.put(uploadUrl, uploadFile, { headers: { 'Content-Type': uploadFile.type } });

      // create note referencing the uploaded file
      await apiClient.post('/notes', { title: uploadFile.name, originalImageUrl: key });
      await fetchAll();
      setUploadFile(null);
    } catch (e: any) {
      setError(e?.response?.data?.message || e?.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <Layout>
      <div className="mb-4 flex items-center gap-4">
        <input className="px-2 py-1 rounded bg-black/40" placeholder="Search" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} />
        <div className="flex gap-2">
          <button onClick={() => setFilterType('all')} className={`px-2 py-1 rounded ${filterType === 'all' ? 'bg-violet-500 text-white' : 'bg-white/5'}`}>All</button>
          <button onClick={() => setFilterType('notes')} className={`px-2 py-1 rounded ${filterType === 'notes' ? 'bg-violet-500 text-white' : 'bg-white/5'}`}>Notes</button>
          <button onClick={() => setFilterType('flashcards')} className={`px-2 py-1 rounded ${filterType === 'flashcards' ? 'bg-violet-500 text-white' : 'bg-white/5'}`}>Flashcards</button>
          <button onClick={() => setFilterType('sessions')} className={`px-2 py-1 rounded ${filterType === 'sessions' ? 'bg-violet-500 text-white' : 'bg-white/5'}`}>Sessions</button>
        </div>
      </div>

      {loading && <div className="p-4">Loading library...</div>}
      {error && <div className="p-3 bg-red-500/10 border border-red-500 text-red-400 rounded mb-4">{error}</div>}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <Card>
          <div className="flex flex-col gap-3">
            <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Upload</div>
            <div className="text-lg font-semibold">Upload study material</div>
            <p className="text-sm text-gray-400">Upload images or PDFs; the file will be stored in S3 and a note will be created.</p>
            <input type="file" accept="image/*,application/pdf" onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)} />
            <div className="flex gap-2 mt-2">
              <button onClick={handleUpload} disabled={!uploadFile || uploading} className="px-3 py-1.5 rounded bg-violet-500 text-white">
                {uploading ? 'Uploading...' : 'Upload'}
              </button>
              <button onClick={() => { setUploadFile(null); setError(null); }} className="px-3 py-1.5 rounded bg-white/5">Clear</button>
            </div>
          </div>
        </Card>
        <Card>
          <div>
            <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Items</div>
            <div className="text-lg font-semibold">Search results</div>
            <div className="mt-3 flex flex-col gap-2 max-h-96 overflow-y-auto">
              {filtered().map((it, idx) => (
                <div key={it.id ?? idx} className="p-3 border rounded bg-white/5">
                  <div className="font-medium">{it.title || it.question || ('Session ' + (it.id ?? idx))}</div>
                  <div className="text-xs text-gray-400">{it.created_at || it.createdAt || ''}</div>
                </div>
              ))}
              {!filtered().length && <div className="text-xs text-gray-500">No items found.</div>}
            </div>
          </div>
        </Card>
        <Card>
          <div>
            <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Stats</div>
            <div className="text-lg font-semibold">Counts</div>
            <div className="mt-3 text-sm text-gray-400">
              <div>Notes: {notes.length}</div>
              <div>Flashcards: {flashcards.length}</div>
              <div>Study sessions: {sessions.length}</div>
            </div>
          </div>
        </Card>
      </div>
    </Layout>
  );
}

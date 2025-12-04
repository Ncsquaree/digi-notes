import Layout from '../components/Layout';
import Card from '../components/Card';
import Button from '../components/Button';
import { useEffect, useRef, useState } from 'react';
import apiClient from '../api/client';
import axios from 'axios';

type Mode = 'quiz' | 'flashcards' | 'mindmap';

export default function Tools() {
  const [inputMode, setInputMode] = useState<'text' | 'image'>('text');
  const [mode, setMode] = useState<Mode>('quiz');
  const [input, setInput] = useState('');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [parsedContent, setParsedContent] = useState<any | null>(null);
  const [results, setResults] = useState<any | null>(null);
  const [processingStatus, setProcessingStatus] = useState<any | null>(null);
  const [noteId, setNoteId] = useState<string | null>(null);

  const pollingRef = useRef<number | null>(null);
  const POLL_MS = Number(import.meta.env.VITE_POLLING_INTERVAL_MS ?? 2000);

  function formatApiError(err: any) {
    if (!err) return 'Unknown error';
    const data = err.response?.data;
    if (data?.error) return String(data.error);
    if (data?.details) return String(data.details);
    if (err.message) return String(err.message);
    return 'API error';
  }

  useEffect(() => {
    return () => {
      stopPolling();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function startPolling(targetNoteId: string) {
    stopPolling();
    pollingRef.current = window.setInterval(async () => {
      try {
        const resp = await apiClient.get(`/notes/${targetNoteId}/status`);
        const data = resp.data || {};
        setProcessingStatus(data);
        if (data.processing_status === 'completed' || data.processing_status === 'partial_success') {
          // terminal state — stop polling and attempt to extract parsed content
          stopPolling();
          let parsed = data.result?.parsed_content ?? data.parsed_content ?? null;
          if (!parsed) {
            // try fetching the note in case parsed content was persisted to the note
            try {
              const noteResp = await apiClient.get(`/notes/${targetNoteId}`);
              const noteData = noteResp.data || {};
              parsed = noteData.note?.parsed_content ?? noteData.parsed_content ?? null;
            } catch (innerErr: any) {
              // ignore — we'll surface that parsed content is missing
            }
          }
          if (!parsed) {
            setLoading(false);
            setParsedContent(null);
            setResults(null);
            setError('Processing completed but no parsed content was attached.');
            return;
          }
          setParsedContent(parsed);
          // after parsing, generate tool-specific outputs (caller is responsible for loading)
          try {
            await generateFromParsed(parsed);
          } catch (genErr: any) {
            setError(formatApiError(genErr));
          }
          setLoading(false);
        } else if (data.processing_status === 'failed') {
          stopPolling();
          setLoading(false);
          setError(data.error || 'Processing failed');
        }
      } catch (err: any) {
        // If the error is a client error (4xx) it's unlikely to recover — stop polling
        const status = err?.response?.status;
        if (status && status >= 400 && status < 500) {
          stopPolling();
          setLoading(false);
          setError(formatApiError(err));
          return;
        }
        // otherwise, surface transient error but keep polling
        setError(formatApiError(err) || 'Status check failed');
      }
    }, POLL_MS);
  }

  function stopPolling() {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }

  async function handleImageUpload() {
    if (!uploadFile) {
      setError('No file selected');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const presignResp = await apiClient.post('/s3/presign-upload', { filename: uploadFile.name, contentType: uploadFile.type });
      const presignData = presignResp.data || {};
      const uploadUrl = presignData.uploadUrl;
      const key = presignData.key;
      if (!uploadUrl || !key) throw new Error('Invalid presign response');

      // upload to S3 directly
      await axios.put(uploadUrl, uploadFile, { headers: { 'Content-Type': uploadFile.type } });

      // create a note referencing the uploaded file
      const noteResp = await apiClient.post('/notes', { title: uploadFile.name, originalImageUrl: key });
      const noteData = noteResp.data || {};
      const createdNoteId = noteData.note?.id;
      if (!createdNoteId) {
        setLoading(false);
        setError('Failed to create note for uploaded file');
        return;
      }
      setNoteId(createdNoteId);

      // trigger processing
      const procResp = await apiClient.post(`/notes/${createdNoteId}/process`);
      const procData = procResp.data || {};
      setProcessingStatus(procData);
      // start polling — polling will clear loading when complete or on terminal error
      startPolling(createdNoteId as string);
    } catch (err: any) {
      setError(formatApiError(err));
      setLoading(false);
    }
  }

  async function generateFromParsed(parsed: any) {
    setError(null);
    try {
      if (mode === 'quiz') {
        const gen = await apiClient.post('/ai/tools/generate-quiz', { content: parsed, questionCount: 10 });
        const quiz = gen.data.quiz;
        setResults({ quiz });
      } else if (mode === 'flashcards') {
        const body: any = { content: parsed, count: 10 };
        if (noteId) body.noteId = noteId;
        const gen = await apiClient.post('/ai/flashcards/generate', body);
        const flashcards = gen.data.flashcards;
        setResults({ flashcards });
      } else if (mode === 'mindmap') {
        const gen = await apiClient.post('/ai/tools/generate-mindmap', { content: parsed });
        const mindmap = gen.data.mindmap;
        setResults({ mindmap });
      }
    } catch (err: any) {
      setError(formatApiError(err));
      throw err;
    }
  }

  async function handleTextGenerate() {
    if (!input?.trim()) {
      setError('Please provide some text to parse');
      return;
    }
    setError(null);
    // The backend's semantic parse endpoint is currently a 501 stub.
    // Use a minimal fallback parsedContent wrapper and call generation endpoints directly.
    setLoading(true);
    setResults(null);
    try {
      // Fallback parsed content — simple wrapper around raw_text
      const parsed = { raw_text: input };
      setParsedContent(parsed);
      await generateFromParsed(parsed);
    } catch (err: any) {
      setError(formatApiError(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleSaveFlashcards() {
    if (!results?.flashcards?.length) {
      setError('No flashcards to save');
      return;
    }
    // If noteId is present and we included it in the generation request, assume backend persisted them.
    if (noteId) {
      // No-op save — already persisted server-side
      return;
    }
    setLoading(true);
    setError(null);
    try {
      // backend may accept bulk creation at POST /flashcards
      const payload = { flashcards: results.flashcards.map((f: any) => ({ question: f.question, answer: f.answer, difficulty: f.difficulty ?? 0 })) };
      await apiClient.post('/flashcards', payload);
    } catch (err: any) {
      setError(formatApiError(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <Layout>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <Card>
          <div className="flex flex-col gap-3">
            <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Tools</div>
            <div className="text-lg font-semibold">AI-ready tools</div>
            <p className="text-xs text-gray-400">Select input mode and tool.</p>
            <div className="flex gap-2 mt-2">
              <button onClick={() => setInputMode('text')} className={`px-3 py-1 rounded ${inputMode === 'text' ? 'bg-violet-500 text-white' : 'bg-white/5'}`}>Text Input</button>
              <button onClick={() => setInputMode('image')} className={`px-3 py-1 rounded ${inputMode === 'image' ? 'bg-violet-500 text-white' : 'bg-white/5'}`}>Image Upload</button>
            </div>
            <div className="flex flex-wrap gap-2 mt-2">
              <button onClick={() => setMode('quiz')} className={`text-[11px] px-3 py-1.5 rounded-full ${mode === 'quiz' ? 'bg-violet-500 text-white' : 'bg-white/5 text-gray-200'}`}>Quiz</button>
              <button onClick={() => setMode('flashcards')} className={`text-[11px] px-3 py-1.5 rounded-full ${mode === 'flashcards' ? 'bg-violet-500 text-white' : 'bg-white/5 text-gray-200'}`}>Flashcards</button>
              <button onClick={() => setMode('mindmap')} className={`text-[11px] px-3 py-1.5 rounded-full ${mode === 'mindmap' ? 'bg-violet-500 text-white' : 'bg-white/5 text-gray-200'}`}>Mindmap</button>
            </div>
          </div>
        </Card>

        <Card>
          <div className="flex flex-col gap-3 h-full">
            <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Input</div>
            {inputMode === 'text' ? (
              <>
                <textarea
                  className="flex-1 w-full bg-black/40 border border-white/10 rounded-xl px-3 py-3 text-xs outline-none focus:border-violet-400 resize-none"
                  placeholder="Paste any chapter content, notes or textbook paragraphs here..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                />
                <div className="flex gap-2 justify-end">
                  <Button type="button" onClick={handleTextGenerate} className="px-3 py-1.5" disabled={loading}>{loading ? 'Generating...' : 'Generate'}</Button>
                </div>
              </>
            ) : (
              <>
                <input type="file" accept="image/*,application/pdf" onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)} />
                <div className="flex gap-2 justify-end">
                  <Button type="button" onClick={handleImageUpload} disabled={!uploadFile || loading}>{loading ? 'Processing...' : 'Upload & Process'}</Button>
                </div>
              </>
            )}
            {error && <div className="p-2 bg-red-500/10 border border-red-500 text-red-400 rounded mt-2">{error}</div>}
            {processingStatus && (
              <div className="mt-2">
                <div className="text-xs text-gray-400">Processing: {processingStatus.processing_status}</div>
                <div className="w-full bg-white/5 rounded h-2 mt-1">
                  <div className="h-2 bg-violet-500 rounded" style={{ width: `${processingStatus.progress_pct ?? 0}%` }} />
                </div>
                <div className="text-xs text-gray-400 mt-1">{processingStatus.current_step}</div>
              </div>
            )}
          </div>
        </Card>

        <Card>
          <div className="flex flex-col gap-3 h-full">
            <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Preview</div>
            <div className="text-sm font-semibold mb-1">{mode === 'quiz' ? 'Quiz preview' : mode === 'flashcards' ? 'Flashcards preview' : 'Mindmap preview'}</div>
            <div className="mt-2">
              {loading && <div className="text-xs text-gray-400">Working...</div>}
              {!loading && results?.quiz && <QuizPreview quiz={results.quiz} />}
              {!loading && results?.flashcards && <FlashcardsPreview flashcards={results.flashcards} onSave={handleSaveFlashcards} saving={loading} />}
              {!loading && results?.mindmap && <MindmapPreview mindmap={results.mindmap} />}
              {!loading && !results && <div className="text-xs text-gray-400">No results yet.</div>}
            </div>
          </div>
        </Card>
      </div>
    </Layout>
  );
}

function QuizPreview({ quiz }: { quiz: any }) {
  if (!quiz?.questions?.length) return <div className="text-xs text-gray-400">No quiz questions generated.</div>;
  return (
    <div className="flex flex-col gap-3">
      {quiz.questions.map((q: any, i: number) => (
        <div key={i} className="p-2 border rounded bg-white/5">
          <div className="font-medium">{i + 1}. {q.text}</div>
          {q.options?.length ? (
            <ul className="text-sm mt-1">
              {q.options.map((opt: any, idx: number) => (
                <li key={idx} className={`py-0.5 ${opt === q.correct ? 'font-semibold' : ''}`}>{opt}</li>
              ))}
            </ul>
          ) : <div className="text-xs text-gray-400 mt-1">Short answer</div>}
        </div>
      ))}
    </div>
  );
}

function FlashcardsPreview({ flashcards, onSave, saving }: { flashcards: any[], onSave: () => void, saving: boolean }) {
  if (!flashcards?.length) return <div className="text-xs text-gray-400">No flashcards generated.</div>;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex justify-end">
        <button onClick={onSave} disabled={saving} className="px-3 py-1 rounded bg-violet-500 text-white">{saving ? 'Saving...' : 'Save Flashcards'}</button>
      </div>
      {flashcards.map((f: any, i: number) => (
        <div key={i} className="p-2 border rounded bg-white/5">
          <div className="font-medium">Q: {f.question}</div>
          <div className="text-sm text-gray-300 mt-1">A: {f.answer}</div>
        </div>
      ))}
    </div>
  );
}

function MindmapPreview({ mindmap }: { mindmap: any }) {
  if (!mindmap) return <div className="text-xs text-gray-400">No mindmap generated.</div>;
  // Simple nested rendering based on nodes/edges or hierarchical nodes
  if (mindmap.nodes) {
    return (
      <div className="flex flex-col gap-2">
        {mindmap.nodes.map((n: any, i: number) => (
          <div key={i} className="p-2 border rounded bg-white/5">
            <div className="font-medium">{n.label || n.title || `Node ${i + 1}`}</div>
            {n.children && (
              <div className="pl-3 mt-1 text-sm text-gray-300">
                {n.children.map((c: any, ci: number) => <div key={ci}>- {c.label || c.title}</div>)}
              </div>
            )}
          </div>
        ))}
      </div>
    );
  }
  return <div className="text-xs text-gray-400">Unsupported mindmap format.</div>;
}

import { FormEvent, useEffect, useState, useCallback } from 'react';
import Layout from '../components/Layout';
import Card from '../components/Card';
import Button from '../components/Button';
import Input from '../components/Input';
import { useAuth } from '../context/AuthContext';
import apiClient from '../api/client';

interface Subject {
  id: string;
  name: string;
  description?: string;
  color?: string;
  icon?: string;
}

interface Chapter {
  id: string;
  subject_id: string;
  name: string;
  description?: string;
  order_index?: number;
}

interface NoteItem {
  id: string;
  chapter_id: string | null;
  title: string;
  parsed_content?: string | null;
}

export default function Subjects() {
  const { user } = useAuth();
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [newSubject, setNewSubject] = useState('');
  const [selectedSubjectId, setSelectedSubjectId] = useState<string | null>(null);
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [chapterTitle, setChapterTitle] = useState('');
  const [chapterClass, setChapterClass] = useState('');
  const [chapterContent, setChapterContent] = useState('');
  const [activeChapterId, setActiveChapterId] = useState<string | null>(null);
  const [activeNoteId, setActiveNoteId] = useState<string | null>(null);

  const [subjectsLoading, setSubjectsLoading] = useState(false);
  const [chaptersLoading, setChaptersLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSubjects = useCallback(async () => {
    if (!user) return;
    setError(null);
    setSubjectsLoading(true);
    try {
      const resp = await apiClient.get('/subjects');
      const data = resp.data?.data ?? resp.data ?? [];
      setSubjects(Array.isArray(data) ? data : []);
      if (!selectedSubjectId && Array.isArray(data) && data.length > 0) setSelectedSubjectId(data[0].id);
    } catch (e: any) {
      setError(e?.response?.data?.message || e?.message || 'Failed to load subjects');
    } finally {
      setSubjectsLoading(false);
    }
  }, [user, selectedSubjectId]);

  const fetchChapters = useCallback(async (subjectId?: string | null) => {
    if (!user || !subjectId) return setChapters([]);
    setError(null);
    setChaptersLoading(true);
    try {
      const resp = await apiClient.get(`/subjects/${subjectId}/chapters`);
      const data = resp.data?.data ?? resp.data ?? [];
      setChapters(Array.isArray(data) ? data : []);
      if (!activeChapterId && Array.isArray(data) && data.length > 0) {
        const first = data[0];
        setActiveChapterId(first.id);
        setChapterTitle(first.name || '');
        setChapterClass(first.description || '');
      }
    } catch (e: any) {
      setError(e?.response?.data?.message || e?.message || 'Failed to load chapters');
    } finally {
      setChaptersLoading(false);
    }
  }, [user, activeChapterId]);

  const fetchChapterNotes = useCallback(async (chapterId?: string | null) => {
    if (!user || !chapterId) {
      setActiveNoteId(null);
      setChapterContent('');
      return;
    }
    setError(null);
    try {
      const resp = await apiClient.get('/notes', { params: { chapterId } });
      const notes: NoteItem[] = resp.data?.notes ?? resp.data?.data ?? resp.data ?? [];
      if (Array.isArray(notes) && notes.length > 0) {
        const n = notes[0];
        setActiveNoteId(n.id);
        setChapterContent(n.parsed_content || '');
      } else {
        setActiveNoteId(null);
        setChapterContent('');
      }
    } catch (e: any) {
      setError(e?.response?.data?.message || e?.message || 'Failed to load notes');
    }
  }, [user]);

  // initial load + polling
  useEffect(() => {
    if (!user) return;
    fetchSubjects();
    const poll = setInterval(fetchSubjects, 30000);
    return () => clearInterval(poll);
  }, [user, fetchSubjects]);

  // when subject changes, load chapters (and poll chapters)
  useEffect(() => {
    if (!user) return;
    if (!selectedSubjectId) {
      setChapters([]);
      return;
    }
    fetchChapters(selectedSubjectId);
    const poll = setInterval(() => fetchChapters(selectedSubjectId), 30000);
    return () => clearInterval(poll);
  }, [user, selectedSubjectId, fetchChapters]);

  // when active chapter changes, load its notes
  useEffect(() => {
    fetchChapterNotes(activeChapterId);
  }, [activeChapterId, fetchChapterNotes]);

  async function handleAddSubject(e: FormEvent) {
    e.preventDefault();
    if (!user || !newSubject.trim()) return;
    setError(null);
    const tempId = `temp-${Date.now()}`;
    const temp: Subject = { id: tempId, name: newSubject.trim() } as Subject;
    const prev = subjects;
    setSubjects([temp, ...subjects]);
    setNewSubject('');
    try {
      const resp = await apiClient.post('/subjects', { name: temp.name });
      const created = resp.data?.data ?? resp.data;
      // replace temp
      setSubjects((cur) => cur.map((s) => (s.id === tempId ? created : s)));
    } catch (e: any) {
      setSubjects(prev);
      setError(e?.response?.data?.message || e?.message || 'Failed to add subject');
    }
  }

  async function handleDeleteSubject(id: string) {
    if (!user) return;
    setError(null);
    const prev = subjects;
    setSubjects((s) => s.filter((x) => x.id !== id));
    if (selectedSubjectId === id) {
      setSelectedSubjectId(null);
      setChapters([]);
      setActiveChapterId(null);
    }
    try {
      await apiClient.delete(`/subjects/${id}`);
    } catch (e: any) {
      setSubjects(prev);
      setError(e?.response?.data?.message || e?.message || 'Failed to delete subject');
    }
  }

  function handleSelectChapterItem(ch: Chapter) {
    setActiveChapterId(ch.id);
    setChapterTitle(ch.name || '');
    setChapterClass(ch.description || '');
  }

  async function handleAddChapter(e: FormEvent) {
    e.preventDefault();
    if (!user || !selectedSubjectId || !chapterTitle.trim()) return;
    setError(null);
    const tempId = `temp-ch-${Date.now()}`;
    const temp: Chapter = { id: tempId, subject_id: selectedSubjectId, name: chapterTitle.trim(), description: chapterClass.trim() } as Chapter;
    const prev = chapters;
    setChapters([temp, ...chapters]);
    setChapterTitle('');
    setChapterClass('');
    setChapterContent('');
    setActiveChapterId(null);
    try {
      const resp = await apiClient.post('/chapters', { subjectId: selectedSubjectId, name: temp.name, description: temp.description });
      const created = resp.data?.data ?? resp.data;
      setChapters((cur) => cur.map((c) => (c.id === tempId ? created : c)));
    } catch (e: any) {
      setChapters(prev);
      setError(e?.response?.data?.message || e?.message || 'Failed to add chapter');
    }
  }

  async function handleSaveChapter() {
    if (!user || !selectedSubjectId || !activeChapterId) return;
    setError(null);
    try {
      // update chapter meta
      await apiClient.put(`/chapters/${activeChapterId}`, { name: chapterTitle, description: chapterClass });

      // fetch or create a note for chapter content
      if (activeNoteId) {
        await apiClient.put(`/notes/${activeNoteId}`, { parsed_content: chapterContent });
      } else {
        const createResp = await apiClient.post('/notes', { chapterId: activeChapterId, title: 'Chapter Notes' });
        const created = createResp.data?.note ?? createResp.data?.data ?? createResp.data;
        const nid = created.id ?? created.note?.id ?? null;
        if (nid) {
          setActiveNoteId(nid);
          await apiClient.put(`/notes/${nid}`, { parsed_content: chapterContent });
        }
      }
    } catch (e: any) {
      setError(e?.response?.data?.message || e?.message || 'Failed to save chapter');
    }
  }

  async function handleDeleteChapter(id: string) {
    if (!user || !selectedSubjectId) return;
    setError(null);
    const prev = chapters;
    setChapters((c) => c.filter((x) => x.id !== id));
    if (activeChapterId === id) {
      setActiveChapterId(null);
      setChapterTitle('');
      setChapterClass('');
      setChapterContent('');
      setActiveNoteId(null);
    }
    try {
      await apiClient.delete(`/chapters/${id}`);
    } catch (e: any) {
      setChapters(prev);
      setError(e?.response?.data?.message || e?.message || 'Failed to delete chapter');
    }
  }

  return (
    <Layout>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card>
          <div className="flex flex-col gap-4">
            <div>
              <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Subjects</div>
              <div className="text-lg font-semibold">Subjects & Classes</div>
              <p className="text-xs text-gray-400 mt-1">
                Create subjects like English, Physics and map them to classes (Class 10, PU2, etc).
              </p>
            </div>
            <form onSubmit={handleAddSubject} className="flex gap-2">
              <Input
                label="New subject"
                value={newSubject}
                onChange={(e) => setNewSubject(e.target.value)}
                placeholder="Eg. English"
              />
              <div className="flex items-end">
                <Button type="submit">Add</Button>
              </div>
            </form>

            {subjectsLoading && <div className="p-4 text-center">Loading subjects...</div>}
            {error && (
              <div className="p-3 bg-red-500/10 border border-red-500 text-red-400 rounded">
                {error}{' '}
                <button
                  onClick={() => {
                    setError(null);
                    fetchSubjects();
                  }}
                  className="underline ml-2"
                >
                  Retry
                </button>
              </div>
            )}

            <div className="max-h-72 overflow-y-auto mt-2 flex flex-col gap-2">
              {subjects.map((s) => (
                <button
                  key={s.id}
                  onClick={() => setSelectedSubjectId(s.id)}
                  className={`flex items-center justify-between px-3 py-2 rounded-xl text-sm border ${
                    selectedSubjectId === s.id
                      ? 'border-violet-500 bg-violet-500/20'
                      : 'border-white/10 bg-white/5 hover:bg-white/10'
                  }`}
                >
                  <span>{s.name}</span>
                  <span
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteSubject(s.id);
                    }}
                    className="text-xs text-gray-400 hover:text-red-400 cursor-pointer"
                  >
                    Delete
                  </span>
                </button>
              ))}
              {!subjects.length && !subjectsLoading && (
                <div className="text-xs text-gray-500 mt-2">No subjects yet. Add your first subject above.</div>
              )}
            </div>
          </div>
        </Card>
        <Card>
          <div className="flex flex-col gap-3 h-full">
            <div>
              <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Chapters</div>
              <div className="text-lg font-semibold">Chapters for selected subject</div>
              <p className="text-xs text-gray-400 mt-1">
                Create chapters and connect them to a particular class. Eg: Class 10 - Prose 1.
              </p>
            </div>
            <form onSubmit={handleAddChapter} className="flex flex-col gap-3">
              <Input
                label="Class"
                value={chapterClass}
                onChange={(e) => setChapterClass(e.target.value)}
                placeholder="Eg. Class 10"
              />
              <Input
                label="Chapter title"
                value={chapterTitle}
                onChange={(e) => setChapterTitle(e.target.value)}
                placeholder="Eg. Ch 2 - A Letter to God"
              />
              <Button type="submit">Add / Save new chapter</Button>
            </form>

            {chaptersLoading && <div className="p-4 text-center">Loading chapters...</div>}

            <div className="flex-1 overflow-y-auto mt-3 flex flex-col gap-2">
              {chapters.map((c) => (
                <button
                  key={c.id}
                  onClick={() => handleSelectChapterItem(c)}
                  className={`flex items-center justify-between px-3 py-2 rounded-xl text-xs border ${
                    activeChapterId === c.id
                      ? 'border-violet-500 bg-violet-500/20'
                      : 'border-white/10 bg-white/5 hover:bg-white/10'
                  }`}
                >
                  <span className="text-left">
                    <div className="font-medium">{c.name}</div>
                    <div className="text-[10px] text-gray-400">{c.description}</div>
                  </span>
                  <span
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteChapter(c.id);
                    }}
                    className="text-[10px] text-gray-400 hover:text-red-400 cursor-pointer"
                  >
                    Delete
                  </span>
                </button>
              ))}
              {!chapters.length && !chaptersLoading && (
                <div className="text-xs text-gray-500 mt-2">
                  No chapters yet for this subject. Use the form above to add one.
                </div>
              )}
            </div>
          </div>
        </Card>
        <Card>
          <div className="flex flex-col gap-3 h-full">
            <div>
              <div className="text-xs uppercase tracking-[0.2em] text-gray-500">Notes</div>
              <div className="text-lg font-semibold">Chapter workspace</div>
              <p className="text-xs text-gray-400 mt-1">
                Write key points, summaries or answers here. It auto-saves to the current chapter.
              </p>
            </div>
            <textarea
              className="flex-1 w-full bg-black/40 border border-white/10 rounded-xl px-3 py-3 text-xs outline-none focus:border-violet-400 resize-none"
              placeholder="Type your chapter notes, Q&A, summaries, formulas..."
              value={chapterContent}
              onChange={(e) => setChapterContent(e.target.value)}
            />
            <div className="flex gap-2 justify-end">
              <Button type="button" onClick={handleSaveChapter}>
                Save changes
              </Button>
            </div>
            <div className="text-[10px] text-gray-500">
              Next: we can plug in AI to generate quizzes, flashcards and mindmaps from this chapter.
            </div>
          </div>
        </Card>
      </div>
    </Layout>
  );
}

const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');
const NoteService = require('../services/note.service');
const S3Service = require('../services/s3.service');
const FlashcardService = require('../services/flashcard.service');
const { ValidationError, NotFoundError, ForbiddenError, ConflictError, InternalError } = require('../utils/errors');

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://ai:8000';
const AI_SERVICE_TIMEOUT = parseInt(process.env.AI_SERVICE_TIMEOUT || '600000', 10); // default 10m
const AI_STATUS_POLL_INTERVAL_MS = parseInt(process.env.AI_STATUS_POLL_INTERVAL_MS || '2000', 10);

async function _persistProcessingResults(noteId, userId, aiData) {
  try {
    const data = aiData.result || aiData.result_bundle || aiData || {};

    // parsed content
    const parsed_content = data.parsed_content || data.parsedContent || data.parsed || null;

    // summaries
    const summary_brief = data.summary_brief || data.summaryBrief || (data.summary && data.summary.brief) || null;
    const summary_detailed = data.summary_detailed || data.summaryDetailed || (data.summary && data.summary.detailed) || null;

    // nodes for knowledge graph
    const nodesArray = data.nodes || data.graph_nodes || (data.knowledge_graph && data.knowledge_graph.nodes) || null;

    // flashcards
    const flashcards = data.flashcards || [];

    // Attach parsed content and optionally nodes (atomic in service)
    if (parsed_content || (nodesArray && nodesArray.length)) {
      await NoteService.attachAIResult(noteId, userId, { parsed_content, nodesArray });
    }

    // Update summaries if present
    const summaryFields = {};
    if (summary_brief) summaryFields.summary_brief = summary_brief;
    if (summary_detailed) summaryFields.summary_detailed = summary_detailed;
    if (Object.keys(summaryFields).length) {
      await NoteService.updateNote(noteId, userId, summaryFields);
    }

    // Persist flashcards if provided
    if (Array.isArray(flashcards) && flashcards.length) {
      try {
        const toInsert = flashcards.map(f => ({ question: f.question, answer: f.answer, difficulty: f.difficulty || 0 }));
        await FlashcardService.createFlashcardsFromAI(noteId, userId, toInsert);
      } catch (fcErr) {
        logger.logError(fcErr, { noteId, userId, task: 'persist_flashcards' });
      }
    }

    // Finalize: mark completed and clear task id
    try {
      await NoteService.updateNote(noteId, userId, { processing_status: 'completed', error_message: null, processing_task_id: null });
    } catch (stErr) {
      logger.logError(stErr, { noteId, userId, fn: '_final_update' });
    }

    return true;
  } catch (err) {
    logger.logError(err, { noteId, userId, fn: '_persistProcessingResults' });
    throw err;
  }
}

async function _pollAiStatusAndPersist(taskId, noteId, userId) {
  const maxAttempts = Math.max(1, Math.floor(AI_SERVICE_TIMEOUT / AI_STATUS_POLL_INTERVAL_MS));
  let attempts = 0;
  try {
    while (attempts < maxAttempts) {
      attempts += 1;
      try {
        // use the taskId as a stable trace id for background polling
        const requestId = taskId;
        const resp = await axios.get(`${AI_SERVICE_URL}/process-note/status/${taskId}`, { timeout: AI_SERVICE_TIMEOUT, headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId } });
        const aiData = resp.data || {};
        const taskNode = aiData.task || {};
        const status = (aiData.status || aiData.state || taskNode.status || taskNode.state || '').toLowerCase();
        if (status === 'completed' || status === 'failed' || status === 'partial_success' || status === 'partial-success') {
          // Terminal state: persist or mark failed
          if (status === 'failed') {
            const errMsg = aiData.error_message || aiData.error || aiData.details || taskNode.error_message || taskNode.error || taskNode.details || 'AI processing failed';
            await NoteService.updateNoteProcessingStatus(noteId, 'failed', errMsg);
            logger.info('ai_processing_failed', { noteId, taskId, userId, attempts });
            return aiData;
          }

          // For completed or partial_success, persist whatever data available
          try {
            await _persistProcessingResults(noteId, userId, aiData);
            logger.info('ai_processing_completed', { noteId, taskId, userId, status, attempts });
          } catch (persistErr) {
            logger.logError(persistErr, { noteId, taskId, userId });
            await NoteService.updateNoteProcessingStatus(noteId, 'failed', 'Failed to persist AI results');
          }
          return aiData;
        }
      } catch (err) {
        // transient network/AI errors - log and continue polling
        logger.logError(err, { noteId, taskId, userId, attempt: attempts });
      }
      await new Promise(r => setTimeout(r, AI_STATUS_POLL_INTERVAL_MS));
    }
    // Timeout
    const msg = `AI status polling timed out after ${attempts} attempts`;
    logger.warn('ai_status_poll_timeout', { noteId, taskId, userId, attempts });
    await NoteService.updateNoteProcessingStatus(noteId, 'failed', msg);
    return null;
  } catch (err) {
    logger.logError(err, { noteId, taskId, userId });
    await NoteService.updateNoteProcessingStatus(noteId, 'failed', 'Unexpected error while polling AI status');
    return null;
  }
}

module.exports = {
  async listNotes(req, res, next) {
    try {
      const userId = req.user.id;
      const opts = { limit: req.query.limit ? Number(req.query.limit) : undefined, offset: req.query.offset ? Number(req.query.offset) : undefined, filters: {} };
      if (req.query.subjectId) opts.filters.subject_id = req.query.subjectId;
      if (req.query.chapterId) opts.filters.chapter_id = req.query.chapterId;
      if (req.query.status) opts.filters.processing_status = req.query.status;
      const notes = await NoteService.getNotesByUserId(userId, opts);
      // support service returning { rows, pagination } or direct array
      const rows = (notes && notes.rows) ? notes.rows : notes;
      const pagination = (notes && notes.pagination) ? notes.pagination : { page: 1, pageSize: Array.isArray(rows) ? rows.length : 0, total: Array.isArray(rows) ? rows.length : 0 };
      return res.json({ success: true, notes: rows, pagination });
    } catch (err) {
      next(err);
    }
  },

  async getNote(req, res, next) {
    try {
      const userId = req.user.id;
      const noteId = req.params.id;
      const note = await NoteService.getNoteById(noteId, userId);
      if (!note) throw new NotFoundError('Note not found');
      if (note.user_id && note.user_id !== userId) throw new ForbiddenError('Access denied');
      return res.json({ success: true, note });
    } catch (err) {
      next(err);
    }
  },

  async createNote(req, res, next) {
    try {
      const userId = req.user.id;
      const payload = { subject_id: req.body.subjectId || null, chapter_id: req.body.chapterId || null, title: req.body.title, original_image_url: req.body.originalImageUrl };
      const created = await NoteService.createNote(userId, payload);
      return res.status(201).json({ success: true, note: created });
    } catch (err) {
      next(err);
    }
  },

  async updateNote(req, res, next) {
    try {
      const userId = req.user.id;
      const noteId = req.params.id;
      const fields = {};
      if (typeof req.body.title !== 'undefined') fields.title = req.body.title;
      if (typeof req.body.subjectId !== 'undefined') fields.subject_id = req.body.subjectId;
      if (typeof req.body.chapterId !== 'undefined') fields.chapter_id = req.body.chapterId;
      const updated = await NoteService.updateNote(noteId, userId, fields);
      return res.json({ success: true, note: updated });
    } catch (err) {
      next(err);
    }
  },

  async deleteNote(req, res, next) {
    try {
      const userId = req.user.id;
      const noteId = req.params.id;
      await NoteService.deleteNote(noteId, userId);
      return res.json({ success: true });
    } catch (err) {
      next(err);
    }
  },

  async processNote(req, res, next) {
    try {
      const userId = req.user.id;
      const noteId = req.params.id;

      const note = await NoteService.getNoteById(noteId, userId);
      if (!note) throw new NotFoundError('Note not found');

      if (!note.original_image_url) {
        throw new ValidationError('original_image_url missing on note');
      }

      if (note.processing_status === 'processing') {
        throw new ConflictError('Note is already being processed');
      }

      // extract s3 key from the URL (strip query params)
      let s3_key = note.original_image_url.split('?')[0];

      // mark processing (support legacy `markProcessing` helper used in tests)
      const taskId = uuidv4();
      let markResult = null;
      try {
        if (typeof NoteService.markProcessing === 'function') {
          markResult = await NoteService.markProcessing(noteId, taskId);
        } else {
          await NoteService.updateNoteProcessingStatus(noteId, 'processing', null);
        }
      } catch (err) {
        logger.logError(err, { noteId, userId, fn: 'mark_processing' });
      }

      // persist task id on the note so frontend/backend can correlate
      try {
        await NoteService.updateNote(noteId, userId, { processing_task_id: taskId });
      } catch (err) {
        logger.logError(err, { noteId, userId, fn: 'persist_task_id' });
      }

      const payload = { task_id: taskId, user_id: userId, note_id: noteId, s3_key, options: req.body.options || {}, await_result: false };

      // Fire-and-forget: POST to AI service and then poll status in background
      const requestId = req.id || (req.headers && req.headers['x-request-id']) || taskId;
      axios.post(`${AI_SERVICE_URL}/process-note`, payload, { timeout: AI_SERVICE_TIMEOUT, headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId } })
        .then(() => {
          // start polling asynchronously (do not await)
          (async () => {
            try {
              await _pollAiStatusAndPersist(taskId, noteId, userId);
            } catch (err) {
              logger.logError(err, { noteId, taskId, userId, fn: 'background_poll' });
              try { await NoteService.updateNoteProcessingStatus(noteId, 'failed', 'Background processing error'); } catch(e){}
            }
          })();
        })
        .catch(async (err) => {
          // immediate failure to call AI service
          logger.logError(err, { noteId, taskId, userId, fn: 'init_ai_call' });
          const msg = err.response ? (err.response.data && err.response.data.error) || 'AI service error' : err.message;
          try { await NoteService.updateNoteProcessingStatus(noteId, 'failed', msg); } catch(e){}
        });
      
      

      // Return canonical response expected by callers/tests
      const data = markResult || { processing_task_id: taskId };
      return res.status(202).json({ success: true, data });
    } catch (err) {
      next(err);
    }
  },

  async getProcessingStatus(req, res, next) {
    try {
      const userId = req.user.id;
      const noteId = req.params.id;
      // Prefer service helper if available (tests expect NoteService.getProcessingStatus)
      if (typeof NoteService.getProcessingStatus === 'function') {
        const status = await NoteService.getProcessingStatus(noteId, userId);
        return res.json({ success: true, status });
      }

      const note = await NoteService.getNoteById(noteId, userId);
      if (!note) throw new NotFoundError('Note not found');

      // If processing and we have a task id, fetch AI live status and merge
      if (note.processing_status === 'processing' && note.processing_task_id) {
        try {
          const requestId = req.id || (req.headers && req.headers['x-request-id']) || note.processing_task_id;
          const resp = await axios.get(`${AI_SERVICE_URL}/process-note/status/${note.processing_task_id}`, { timeout: Math.min(30000, AI_SERVICE_TIMEOUT), headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId } });
          const aiData = resp.data || {};
          const taskNode = aiData.task || {};
          return res.json({
            success: true,
            note_id: noteId,
            processing_status: note.processing_status,
            error_message: note.error_message,
            ai_status: aiData.status || aiData.state || taskNode.status || taskNode.state || null,
            progress_pct: aiData.progress_pct || aiData.progress || taskNode.progress_pct || taskNode.progress || null,
            current_step: aiData.current_step || taskNode.current_step || null,
            steps_completed: aiData.steps_completed || taskNode.steps_completed || [],
            steps_failed: aiData.steps_failed || taskNode.steps_failed || [],
            result: aiData.result || aiData.result_bundle || taskNode.result || null,
          });
        } catch (err) {
          logger.logError(err, { noteId, userId, fn: 'getProcessingStatus' });
          // fallthrough to return DB status
        }
      }

      // Return DB status only
      return res.json({ success: true, note_id: noteId, processing_status: note.processing_status, error_message: note.error_message });
    } catch (err) {
      next(err);
    }
  }
};

const axios = require('axios');
const logger = require('../utils/logger');
const FlashcardService = require('../services/flashcard.service');
const { ValidationError } = require('../utils/errors');

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://ai:8000';
const AI_SERVICE_TIMEOUT = parseInt(process.env.AI_SERVICE_TIMEOUT || '60000', 10);

module.exports = {
  async generateFlashcards(req, res, next) {
    try {
      const requestId = req.id || (req.headers && req.headers['x-request-id']) || null;
      const userId = (req.user && (req.user.id || req.user.userId));
      const content = req.body.content;
      const count = req.body.count;
      const noteId = req.body.noteId;

      if (!content || typeof content !== 'object') {
        throw new ValidationError('content (ParsedContent) is required')
      }

      const payload = { parsed_content: content, count, options: {} };
      const resp = await axios.post(`${AI_SERVICE_URL}/flashcards/generate`, payload, { timeout: AI_SERVICE_TIMEOUT, headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId } });

      const aiData = resp.data || {};
      const flashcards = aiData.flashcards || [];
      let persistedInfo = null;

      if (noteId && Array.isArray(flashcards) && flashcards.length) {
        // map to backend format and bulk insert
        const toInsert = flashcards.map(f => ({ note_id: noteId, user_id: userId, question: f.question, answer: f.answer, difficulty: f.difficulty || 0 }));
        const inserted = await FlashcardService.createFlashcardsFromAI(noteId, userId, toInsert);
        persistedInfo = { persisted: true, persisted_count: inserted.length };
        logger.info('ai_flashcards_generated', { requestId, userId, count: flashcards.length, persisted: true });
        return res.json({ success: true, flashcards: inserted, metadata: { ...aiData.metadata, ...persistedInfo } });
      }

      logger.info('ai_flashcards_generated', { requestId, userId, count: flashcards.length, persisted: false });
      return res.json({ success: true, flashcards, metadata: aiData.metadata || {} });
    } catch (err) {
      // axios mapping
      if (err.code === 'ECONNABORTED') {
        return res.status(504).json({ success: false, error: 'AI service timeout', details: err.message });
      }
      if (err.response) {
        const status = err.response.status || 502;
        const aiErr = err.response.data || {};
        const errMsg = aiErr.error || 'AI service error';
        const details = aiErr.details || err.message || aiErr;
        const payload = { success: false, error: errMsg, details };
        if (aiErr.request_id) payload.request_id = aiErr.request_id;
        // prefer header-propagated request id from AI service when present
        if (err.response.headers && err.response.headers['x-request-id']) payload.request_id = payload.request_id || err.response.headers['x-request-id'];
        return res.status(status).json(payload);
      }
      next(err);
    }
  },

  // stubs for other AI proxy endpoints (left as 501 for future phases)
  async processNote(req, res, next) { res.status(501).json({ message: 'Not implemented' }); },
  async extractOCR(req, res, next) { res.status(501).json({ message: 'Not implemented' }); },
  async parseSemantic(req, res, next) { res.status(501).json({ message: 'Not implemented' }); },
  async summarizeContent(req, res, next) { res.status(501).json({ message: 'Not implemented' }); },
  async generateQuiz(req, res, next) {
    try {
      const requestId = req.id || (req.headers && req.headers['x-request-id']) || null;
      const userId = (req.user && (req.user.id || req.user.userId));
      const content = req.body.content;
      const questionCount = req.body.questionCount || req.body.question_count || 10;
      const questionTypes = req.body.questionTypes || req.body.question_types || null;

      if (!content || typeof content !== 'object') {
        throw new ValidationError('content (ParsedContent) is required');
      }

      const payload = { parsed_content: content, question_count: questionCount, question_types: questionTypes, options: req.body.options || {} };
      const resp = await axios.post(`${AI_SERVICE_URL}/tools/generate-quiz`, payload, { timeout: AI_SERVICE_TIMEOUT, headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId } });

      const aiData = resp.data || {};
      const quiz = aiData.quiz || {};

      logger.info('ai_quiz_generated', { requestId, userId, question_count: (quiz.questions || []).length });
      return res.json({ success: true, quiz, metadata: aiData.metadata || {} });
    } catch (err) {
      if (err.code === 'ECONNABORTED') {
        return res.status(504).json({ success: false, error: 'AI service timeout', details: err.message });
      }
      if (err.response) {
        const status = err.response.status || 502;
        const aiErr = err.response.data || {};
        const errMsg = aiErr.error || 'AI service error';
        const details = aiErr.details || err.message || aiErr;
        const payload = { success: false, error: errMsg, details };
        if (aiErr.request_id) payload.request_id = aiErr.request_id;
        if (err.response.headers && err.response.headers['x-request-id']) payload.request_id = payload.request_id || err.response.headers['x-request-id'];
        return res.status(status).json(payload);
      }
      next(err);
    }
  },

  async generateMindmap(req, res, next) {
    try {
      const requestId = req.id || (req.headers && req.headers['x-request-id']) || null;
      const userId = (req.user && (req.user.id || req.user.userId));
      const content = req.body.content;

      if (!content || typeof content !== 'object') {
        throw new ValidationError('content (ParsedContent) is required');
      }

      const payload = { parsed_content: content, options: req.body.options || {} };
      const resp = await axios.post(`${AI_SERVICE_URL}/tools/generate-mindmap`, payload, { timeout: AI_SERVICE_TIMEOUT, headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId } });

      const aiData = resp.data || {};
      const mindmap = aiData.mindmap || {};

      logger.info('ai_mindmap_generated', { requestId, userId, node_count: (mindmap.nodes || []).length, edge_count: (mindmap.edges || []).length });
      return res.json({ success: true, mindmap, metadata: aiData.metadata || {} });
    } catch (err) {
      if (err.code === 'ECONNABORTED') {
        return res.status(504).json({ success: false, error: 'AI service timeout', details: err.message });
      }
      if (err.response) {
        const status = err.response.status || 502;
        const aiErr = err.response.data || {};
        const errMsg = aiErr.error || 'AI service error';
        const details = aiErr.details || err.message || aiErr;
        const payload = { success: false, error: errMsg, details };
        if (aiErr.request_id) payload.request_id = aiErr.request_id;
        if (err.response.headers && err.response.headers['x-request-id']) payload.request_id = payload.request_id || err.response.headers['x-request-id'];
        return res.status(status).json(payload);
      }
      next(err);
    }
  },
  async visualizeGraph(req, res, next) { res.status(501).json({ message: 'Not implemented' }); },
  async getRelatedConcepts(req, res, next) { res.status(501).json({ message: 'Not implemented' }); },
};

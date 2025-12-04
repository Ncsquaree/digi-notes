const express = require('express');
const router = express.Router();
const authenticate = require('../middleware/authenticate');
const { validate } = require('../middleware/validation');
const {
	processNoteValidation,
	ocrExtractValidation,
	semanticParseValidation,
	summarizeValidation,
	generateFlashcardsValidation,
	generateQuizValidation,
	generateMindmapValidation,
	visualizeGraphValidation,
	relatedConceptsValidation,
} = require('../validators/ai.validators');

const aiController = require('../controllers/ai.controller');

/**
 * @openapi
 * tags:
 *   - name: AI
 *     description: AI-related proxy endpoints (process, OCR, semantic parsing, tools)
 */
// All AI routes require authentication
router.use(authenticate);

// These endpoints are proxies/stubs that will call the AI service in a later phase.
/**
 * @openapi
 * /api/ai/process-note:
 *   post:
 *     tags:
 *       - AI
 *     summary: Submit a note for AI processing (OCR, parse, summarize, graph, flashcards)
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       202:
 *         description: Accepted
 */
router.post('/process-note', validate(processNoteValidation), aiController.processNote);

/**
 * @openapi
 * /api/ai/ocr/extract:
 *   post:
 *     tags:
 *       - AI
 *     summary: Extract text from an image via OCR
 *     security:
 *       - bearerAuth: []
 */
router.post('/ocr/extract', validate(ocrExtractValidation), aiController.extractOCR);

/**
 * @openapi
 * /api/ai/parse/semantic:
 *   post:
 *     tags:
 *       - AI
 *     summary: Parse academic content into structured ParsedContent
 *     security:
 *       - bearerAuth: []
 */
router.post('/parse/semantic', validate(semanticParseValidation), aiController.parseSemantic);

/**
 * @openapi
 * /api/ai/summarize:
 *   post:
 *     tags:
 *       - AI
 *     summary: Generate summaries for parsed content
 *     security:
 *       - bearerAuth: []
 */
router.post('/summarize', validate(summarizeValidation), aiController.summarizeContent);

/**
 * @openapi
 * /api/ai/flashcards/generate:
 *   post:
 *     tags:
 *       - AI
 *     summary: Generate flashcards from parsed content
 *     security:
 *       - bearerAuth: []
 */
router.post('/flashcards/generate', validate(generateFlashcardsValidation), aiController.generateFlashcards);

/**
 * @openapi
 * /api/ai/tools/generate-quiz:
 *   post:
 *     tags:
 *       - AI
 *     summary: Generate quiz questions from parsed content
 */
router.post('/tools/generate-quiz', validate(generateQuizValidation), aiController.generateQuiz);

/**
 * @openapi
 * /api/ai/tools/generate-mindmap:
 *   post:
 *     tags:
 *       - AI
 *     summary: Generate a mindmap representation
 */
router.post('/tools/generate-mindmap', validate(generateMindmapValidation), aiController.generateMindmap);

/**
 * @openapi
 * /api/ai/graph/visualize/{userId}:
 *   get:
 *     tags:
 *       - AI
 *     summary: Visualize user's knowledge graph
 */
router.get('/graph/visualize/:userId', validate(visualizeGraphValidation), aiController.visualizeGraph);

/**
 * @openapi
 * /api/ai/graph/related-concepts/{conceptId}:
 *   get:
 *     tags:
 *       - AI
 *     summary: Get related concepts for a concept id
 */
router.get('/graph/related-concepts/:conceptId', validate(relatedConceptsValidation), aiController.getRelatedConcepts);

module.exports = router;

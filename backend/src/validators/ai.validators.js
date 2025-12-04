const { body, param, query } = require('express-validator');

const processNoteValidation = [
  body('noteId').exists().isUUID(4).withMessage('noteId is required and must be a UUID'),
  body('options').optional().isObject().withMessage('options must be an object'),
  body('options.generateFlashcards').optional().isBoolean(),
  body('options.generateSummary').optional().isBoolean(),
  body('options.buildGraph').optional().isBoolean(),
];

const ocrExtractValidation = [body('imageUrl').exists().isURL().isLength({ max: 2048 }).withMessage('imageUrl is required')];

const semanticParseValidation = [body('text').exists().isString().isLength({ max: 50000 }).withMessage('text is required')];

const summarizeValidation = [
  body('content').exists().withMessage('content is required'),
  body('mode').optional().isIn(['brief', 'detailed']).withMessage('mode must be brief or detailed'),
];

const generateFlashcardsValidation = [
  body('content').exists().withMessage('content is required'),
  body('count').optional().isInt({ min: 1, max: 50 }).toInt(),
];

const generateQuizValidation = [
  body('content').exists().withMessage('content is required'),
  body('questionCount').optional().isInt({ min: 1, max: 20 }).toInt(),
  body('questionTypes').optional().isArray(),
];

const generateMindmapValidation = [body('content').exists().withMessage('content is required')];

const visualizeGraphValidation = [
  param('userId').isUUID(4).withMessage('userId must be a UUID'),
  query('depth').optional().isInt({ min: 1, max: 5 }).toInt(),
];

const relatedConceptsValidation = [param('conceptId').exists().isString().withMessage('conceptId is required'), query('limit').optional().isInt({ min: 1, max: 50 }).toInt()];

module.exports = {
  processNoteValidation,
  ocrExtractValidation,
  semanticParseValidation,
  summarizeValidation,
  generateFlashcardsValidation,
  generateQuizValidation,
  generateMindmapValidation,
  visualizeGraphValidation,
  relatedConceptsValidation,
};

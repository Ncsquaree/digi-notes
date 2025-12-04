const { body, param, query } = require('express-validator');

const doEscape = process.env.SANITIZE_HTML === 'true';

let questionChain = body('question').exists().isString().isLength({ max: 2000 }).trim().withMessage('question is required');
if (doEscape) questionChain = questionChain.escape();

let answerChain = body('answer').exists().isString().isLength({ max: 5000 }).trim().withMessage('answer is required');
if (doEscape) answerChain = answerChain.escape();

const createFlashcardValidation = [
  body('noteId').exists().isUUID(4).withMessage('noteId is required and must be a UUID'),
  questionChain,
  answerChain,
];

const updateFlashcardValidation = [
  param('id').isUUID(4).withMessage('id must be a valid UUID'),
  (function () { let c = body('question').optional().isString().isLength({ max: 2000 }).trim(); if (doEscape) c = c.escape(); return c; })(),
  (function () { let c = body('answer').optional().isString().isLength({ max: 5000 }).trim(); if (doEscape) c = c.escape(); return c; })(),
];

const getFlashcardValidation = [param('id').isUUID(4).withMessage('id must be a valid UUID')];
const deleteFlashcardValidation = [param('id').isUUID(4).withMessage('id must be a valid UUID')];

const reviewFlashcardValidation = [
  param('id').isUUID(4).withMessage('id must be a valid UUID'),
  body('quality').exists().isInt({ min: 0, max: 5 }).toInt().withMessage('quality must be 0-5'),
  body('timeSpentSeconds').optional().isInt({ min: 0, max: 3600 }).toInt(),
];

const listFlashcardsValidation = [
  query('noteId').optional().isUUID(4),
  query('page').optional().isInt({ min: 1 }).toInt(),
  query('limit').optional().isInt({ min: 1, max: 100 }).toInt(),
];

const dueFlashcardsValidation = [query('limit').optional().isInt({ min: 1, max: 100 }).toInt()];

module.exports = {
  createFlashcardValidation,
  updateFlashcardValidation,
  getFlashcardValidation,
  deleteFlashcardValidation,
  reviewFlashcardValidation,
  listFlashcardsValidation,
  dueFlashcardsValidation,
};

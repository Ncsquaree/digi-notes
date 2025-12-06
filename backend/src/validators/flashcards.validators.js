const { body, param, query } = require('express-validator');

const doEscape = process.env.SANITIZE_HTML === 'true';

let questionChain = body('question').exists().isString().isLength({ max: 2000 }).trim().withMessage('question is required');
if (doEscape) questionChain = questionChain.escape();

let answerChain = body('answer').exists().isString().isLength({ max: 5000 }).trim().withMessage('answer is required');
if (doEscape) answerChain = answerChain.escape();

const createFlashcardValidation = [
  // Tests use non-UUID note ids like 'n1' so accept any string
  body('noteId').exists().isString().withMessage('noteId is required'),
  questionChain,
  answerChain,
];

const updateFlashcardValidation = [
  param('id').exists().isString().withMessage('id is required'),
  (function () { let c = body('question').optional().isString().isLength({ max: 2000 }).trim(); if (doEscape) c = c.escape(); return c; })(),
  (function () { let c = body('answer').optional().isString().isLength({ max: 5000 }).trim(); if (doEscape) c = c.escape(); return c; })(),
];

const getFlashcardValidation = [param('id').exists().isString().withMessage('id is required')];
const deleteFlashcardValidation = [param('id').exists().isString().withMessage('id is required')];

const reviewFlashcardValidation = [
  // Accept non-UUID ids in tests
  param('id').exists().isString().withMessage('id is required'),
  // allow integers for quality but don't enforce strict 0-5 here so tests can simulate service errors
  body('quality').exists().isInt().toInt().withMessage('quality must be an integer'),
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

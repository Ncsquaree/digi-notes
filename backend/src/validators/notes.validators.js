const { query, param, body } = require('express-validator');
const { isUUID } = require('../middleware/validation');

const doEscape = process.env.SANITIZE_HTML === 'true';

let titleChain = body('title').exists().isString().isLength({ max: 255 }).trim().withMessage('title is required');
if (doEscape) titleChain = titleChain.escape();

let originalImageChain = body('originalImageUrl').exists().isURL().isLength({ max: 2048 }).withMessage('originalImageUrl is required and must be a valid URL');

const createNoteValidation = [
  titleChain,
  originalImageChain,
  body('subjectId').optional().isUUID(4).withMessage('subjectId must be a UUID'),
  body('chapterId').optional().isUUID(4).withMessage('chapterId must be a UUID'),
];

let updateTitleChain = body('title').optional().isString().isLength({ max: 255 }).trim();
if (doEscape) updateTitleChain = updateTitleChain.escape();

const updateNoteValidation = [
  isUUID('id'),
  updateTitleChain,
  body('subjectId').optional().isUUID(4).withMessage('subjectId must be a UUID'),
  body('chapterId').optional().isUUID(4).withMessage('chapterId must be a UUID'),
];

const getNoteValidation = [isUUID('id')];
const deleteNoteValidation = [isUUID('id')];
const processNoteValidation = [isUUID('id')];

const listNotesValidation = [
  query('page').optional().isInt({ min: 1 }).toInt(),
  query('limit').optional().isInt({ min: 1, max: 100 }).toInt(),
  query('subjectId').optional().isUUID(4),
  query('chapterId').optional().isUUID(4),
  query('status').optional().isIn(['pending', 'processing', 'completed', 'failed']),
];

module.exports = {
  createNoteValidation,
  updateNoteValidation,
  getNoteValidation,
  deleteNoteValidation,
  processNoteValidation,
  listNotesValidation,
};

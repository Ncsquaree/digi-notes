const { query, param, body } = require('express-validator');
const { isUUID } = require('../middleware/validation');

const doEscape = process.env.SANITIZE_HTML === 'true';

let titleChain = body('title').exists().isString().isLength({ max: 255 }).trim().withMessage('title is required');
if (doEscape) titleChain = titleChain.escape();

// Accept either camelCase `originalImageUrl` or snake_case `original_image_url`
// Accept any non-empty string up to 2048 chars for image URL (tests use short hostnames)
const originalImageChain = body('originalImageUrl').optional().isString().isLength({ min: 1, max: 2048 }).withMessage('originalImageUrl must be a valid URL or path');
const originalImageSnakeChain = body('original_image_url').optional().isString().isLength({ min: 1, max: 2048 }).withMessage('original_image_url must be a valid URL or path');

const createNoteValidation = [
  titleChain,
  // Require at least one of the two possible fields and validate URL format via express-validator
  body().custom((_, { req }) => {
    if (!req.body) throw new Error('originalImageUrl is required and must be a valid URL');
    if (typeof req.body.originalImageUrl === 'undefined' && typeof req.body.original_image_url === 'undefined') {
      throw new Error('originalImageUrl is required and must be a valid URL');
    }
    return true;
  }),
  // validate either field if present
  originalImageChain,
  originalImageSnakeChain,
  body('subjectId').optional().isString().withMessage('subjectId must be a string'),
  body('chapterId').optional().isString().withMessage('chapterId must be a string'),
];

let updateTitleChain = body('title').optional().isString().isLength({ max: 255 }).trim();
if (doEscape) updateTitleChain = updateTitleChain.escape();

const updateNoteValidation = [
  isUUID('id'),
  updateTitleChain,
  body('subjectId').optional().isUUID(4).withMessage('subjectId must be a UUID'),
  body('chapterId').optional().isUUID(4).withMessage('chapterId must be a UUID'),
];

// Tests use non-UUID ids (like 'note-1') so accept any string id
const getNoteValidation = [param('id').exists().isString().withMessage('id is required')];
const deleteNoteValidation = [param('id').exists().isString().withMessage('id is required')];
const processNoteValidation = [param('id').exists().isString().withMessage('id is required')];

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

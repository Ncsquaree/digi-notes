const { body, param, query } = require('express-validator');

const doEscape = process.env.SANITIZE_HTML === 'true';

let nameRequiredChain = body('name').exists().isString().isLength({ max: 255 }).trim().withMessage('name is required');
if (doEscape) nameRequiredChain = nameRequiredChain.escape();

const createChapterValidation = [
  body('subjectId').optional().isUUID(4).withMessage('subjectId must be a UUID'),
  nameRequiredChain,
  body('description').optional().isString().isLength({ max: 5000 }).trim(),
  body('orderIndex').optional().isInt({ min: 0 }).toInt(),
];

const updateChapterValidation = [
  param('id').isUUID(4).withMessage('id must be a valid UUID'),
  (function () { let c = body('name').optional().isString().isLength({ max: 255 }).trim(); if (doEscape) c = c.escape(); return c; })(),
  body('description').optional().isString().isLength({ max: 5000 }).trim(),
  body('orderIndex').optional().isInt({ min: 0 }).toInt(),
];

const getChapterValidation = [param('id').isUUID(4).withMessage('id must be a valid UUID')];
const deleteChapterValidation = [param('id').isUUID(4).withMessage('id must be a valid UUID')];

const reorderChapterValidation = [
  param('id').isUUID(4).withMessage('id must be a valid UUID'),
  body('orderIndex').exists().isInt({ min: 0 }).toInt().withMessage('orderIndex is required and must be >= 0'),
];

const listChaptersValidation = [
  query('subjectId').optional().isUUID(4),
  query('page').optional().isInt({ min: 1 }).toInt(),
  query('limit').optional().isInt({ min: 1, max: 100 }).toInt(),
];

module.exports = {
  createChapterValidation,
  updateChapterValidation,
  getChapterValidation,
  deleteChapterValidation,
  reorderChapterValidation,
  listChaptersValidation,
};

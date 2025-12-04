const { body, param, query } = require('express-validator');

const colorRegex = /^#[0-9A-Fa-f]{6}$/;

const doEscape = process.env.SANITIZE_HTML === 'true';

let nameChain = body('name').exists().isString().isLength({ max: 255 }).trim().withMessage('name is required');
if (doEscape) nameChain = nameChain.escape();

const createSubjectValidation = [
  nameChain,
  body('description').optional().isString().isLength({ max: 5000 }).trim(),
  body('color').optional().matches(colorRegex).withMessage('color must be a hex string like #RRGGBB'),
  body('icon').optional().isString().isLength({ max: 50 }).withMessage('icon must be at most 50 chars'),
];

const updateSubjectValidation = [
  param('id').isUUID(4).withMessage('id must be a valid UUID'),
  (function () { let c = body('name').optional().isString().isLength({ max: 255 }).trim(); if (doEscape) c = c.escape(); return c; })(),
  body('description').optional().isString().isLength({ max: 5000 }).trim(),
  body('color').optional().matches(colorRegex).withMessage('color must be a hex string like #RRGGBB'),
  body('icon').optional().isString().isLength({ max: 50 }),
];

const getSubjectValidation = [param('id').isUUID(4).withMessage('id must be a valid UUID')];
const deleteSubjectValidation = [param('id').isUUID(4).withMessage('id must be a valid UUID')];

const listSubjectsValidation = [
  query('page').optional().isInt({ min: 1 }).toInt(),
  query('limit').optional().isInt({ min: 1, max: 100 }).toInt(),
];

module.exports = {
  createSubjectValidation,
  updateSubjectValidation,
  getSubjectValidation,
  deleteSubjectValidation,
  listSubjectsValidation,
};

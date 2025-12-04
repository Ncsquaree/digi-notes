const { body } = require('express-validator');

const allowedContentTypes = ['image/jpeg', 'image/png', 'application/pdf'];

const presignUploadValidation = [
  body('filename').exists().isString().trim().notEmpty().withMessage('filename required'),
  body('contentType').exists().isString().isIn(allowedContentTypes).withMessage('Invalid content type')
];

module.exports = { presignUploadValidation };

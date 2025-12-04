const express = require('express');
const router = express.Router();
const authenticate = require('../middleware/authenticate');
const { validate } = require('../middleware/validation');
const { presignUploadValidation } = require('../validators/s3.validators');
const s3Controller = require('../controllers/s3.controller');

router.use(authenticate);

/**
 * @openapi
 * /api/s3/presign-upload:
 *   post:
 *     tags:
 *       - S3
 *     summary: Get a presigned upload URL for S3
 *     security:
 *       - bearerAuth: []
 */
router.post('/presign-upload', validate(presignUploadValidation), s3Controller.getPresignedUploadUrl);

module.exports = router;

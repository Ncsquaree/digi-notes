const express = require('express');
const router = express.Router();
const authenticate = require('../middleware/authenticate');
const { validate } = require('../middleware/validation');
const {
	listChaptersValidation,
	getChapterValidation,
	createChapterValidation,
	updateChapterValidation,
	deleteChapterValidation,
	reorderChapterValidation,
} = require('../validators/chapters.validators');

/**
 * @openapi
 * tags:
 *   - name: Chapters
 *     description: Chapter management under subjects
 */
// All chapter routes require authentication
router.use(authenticate);

const chapterController = require('../controllers/chapter.controller');

/**
 * @openapi
 * /api/chapters:
 *   get:
 *     tags:
 *       - Chapters
 *     summary: List chapters
 */
router.get('/', validate(listChaptersValidation), chapterController.listChapters);

/**
 * @openapi
 * /api/chapters/{id}:
 *   get:
 *     tags:
 *       - Chapters
 *     summary: Get a chapter
 */
router.get('/:id', validate(getChapterValidation), chapterController.getChapter);

/**
 * @openapi
 * /api/chapters:
 *   post:
 *     tags:
 *       - Chapters
 *     summary: Create a chapter
 */
router.post('/', validate(createChapterValidation), chapterController.createChapter);

/**
 * @openapi
 * /api/chapters/{id}:
 *   put:
 *     tags:
 *       - Chapters
 *     summary: Update a chapter
 */
router.put('/:id', validate(updateChapterValidation), chapterController.updateChapter);

/**
 * @openapi
 * /api/chapters/{id}:
 *   delete:
 *     tags:
 *       - Chapters
 *     summary: Delete a chapter
 */
router.delete('/:id', validate(deleteChapterValidation), chapterController.deleteChapter);

/**
 * @openapi
 * /api/chapters/{id}/reorder:
 *   put:
 *     tags:
 *       - Chapters
 *     summary: Reorder a chapter
 */
router.put('/:id/reorder', validate(reorderChapterValidation), chapterController.reorderChapter);

/**
 * @openapi
 * /api/chapters/{id}/notes:
 *   get:
 *     tags:
 *       - Chapters
 *     summary: Get notes for chapter
 */
router.get('/:id/notes', validate(getChapterValidation), chapterController.getChapterNotes);

module.exports = router;

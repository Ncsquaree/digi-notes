const express = require('express');
const router = express.Router();
const notesController = require('../controllers/notes.controller');

router.post('/', notesController.createNote);
router.get('/:id', notesController.getNote);
router.get('/', notesController.listNotes);

module.exports = router;


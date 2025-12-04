const UserService = require('./user.service');
const SubjectService = require('./subject.service');
const ChapterService = require('./chapter.service');
const NoteService = require('./note.service');
const FlashcardService = require('./flashcard.service');
const S3Service = require('./s3.service');

module.exports = {
  UserService,
  SubjectService,
  ChapterService,
  NoteService,
  FlashcardService,
  S3Service,
};

exports.up = (pgm) => {
  pgm.sql("COMMENT ON TABLE flashcards IS 'Fields difficulty, interval, repetitions support SM-2 spaced repetition algorithm (difficulty stored as easiness*10)' ");
  pgm.sql("COMMENT ON COLUMN notes.parsed_content IS 'JSONB structured content produced by LLM parsing'");
};

exports.down = (pgm) => {
  pgm.sql("COMMENT ON TABLE flashcards IS NULL");
  pgm.sql("COMMENT ON COLUMN notes.parsed_content IS NULL");
};

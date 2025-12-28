/* eslint-disable camelcase */
exports.shorthands = undefined;

exports.up = (pgm) => {
  pgm.addColumn('notes', {
    ocr_metadata: { type: 'jsonb' },
  });
  pgm.sql("COMMENT ON COLUMN notes.ocr_metadata IS 'OCR service used, confidence, preprocessing steps';");
};

exports.down = (pgm) => {
  pgm.dropColumn('notes', 'ocr_metadata');
};

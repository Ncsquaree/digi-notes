/* eslint-disable camelcase */
exports.shorthands = undefined;

exports.up = (pgm) => {
  // extensions
  pgm.createExtension('uuid-ossp', { ifNotExists: true });
  pgm.sql("SET timezone = 'UTC'");

  // users
  pgm.createTable('users', {
    id: { type: 'uuid', primaryKey: true, default: pgm.func('uuid_generate_v4()') },
    email: { type: 'varchar(255)', notNull: true },
    password_hash: { type: 'varchar(255)', notNull: true },
    first_name: { type: 'varchar(100)' },
    last_name: { type: 'varchar(100)' },
    created_at: { type: 'timestamp', notNull: true, default: pgm.func('CURRENT_TIMESTAMP') },
    updated_at: { type: 'timestamp', notNull: true, default: pgm.func('CURRENT_TIMESTAMP') },
    last_login: { type: 'timestamp' },
    is_active: { type: 'boolean', notNull: true, default: true },
    email_verified: { type: 'boolean', notNull: true, default: false }
  });
  pgm.createIndex('users', 'email', { name: 'idx_users_email' });
  pgm.createIndex('users', 'created_at', { name: 'idx_users_created_at' });
  pgm.addConstraint('users', 'users_email_unique', { unique: ['email'] });

  // subjects
  pgm.createTable('subjects', {
    id: { type: 'uuid', primaryKey: true, default: pgm.func('uuid_generate_v4()') },
    user_id: { type: 'uuid', notNull: true },
    name: { type: 'varchar(255)', notNull: true },
    description: { type: 'text' },
    color: { type: 'varchar(7)' },
    icon: { type: 'varchar(50)' },
    created_at: { type: 'timestamp', notNull: true, default: pgm.func('CURRENT_TIMESTAMP') },
    updated_at: { type: 'timestamp', notNull: true, default: pgm.func('CURRENT_TIMESTAMP') }
  });
  pgm.createIndex('subjects', 'user_id', { name: 'idx_subjects_user_id' });
  pgm.addConstraint('subjects', 'subjects_user_name_unique', { unique: ['user_id', 'name'] });
  pgm.addConstraint('subjects', 'subjects_user_fk', { foreignKeys: [{ columns: 'user_id', references: 'users(id)', onDelete: 'cascade' }] });

  // chapters
  pgm.createTable('chapters', {
    id: { type: 'uuid', primaryKey: true, default: pgm.func('uuid_generate_v4()') },
    subject_id: { type: 'uuid', notNull: true },
    name: { type: 'varchar(255)', notNull: true },
    description: { type: 'text' },
    order_index: { type: 'integer', notNull: true, default: 0 },
    created_at: { type: 'timestamp', notNull: true, default: pgm.func('CURRENT_TIMESTAMP') },
    updated_at: { type: 'timestamp', notNull: true, default: pgm.func('CURRENT_TIMESTAMP') }
  });
  pgm.createIndex('chapters', 'subject_id', { name: 'idx_chapters_subject_id' });
  pgm.createIndex('chapters', 'order_index', { name: 'idx_chapters_order_index' });
  pgm.addConstraint('chapters', 'chapters_subject_fk', { foreignKeys: [{ columns: 'subject_id', references: 'subjects(id)', onDelete: 'cascade' }] });

  // notes
  pgm.createTable('notes', {
    id: { type: 'uuid', primaryKey: true, default: pgm.func('uuid_generate_v4()') },
    user_id: { type: 'uuid', notNull: true },
    subject_id: { type: 'uuid' },
    chapter_id: { type: 'uuid' },
    title: { type: 'varchar(255)', notNull: true },
    original_image_url: { type: 'text', notNull: true },
    processed_image_url: { type: 'text' },
    ocr_text: { type: 'text' },
    parsed_content: { type: 'jsonb' },
    summary_brief: { type: 'text' },
    summary_detailed: { type: 'text' },
    processing_status: { type: 'varchar(50)', notNull: true, default: 'pending' },
    error_message: { type: 'text' },
    created_at: { type: 'timestamp', notNull: true, default: pgm.func('CURRENT_TIMESTAMP') },
    updated_at: { type: 'timestamp', notNull: true, default: pgm.func('CURRENT_TIMESTAMP') },
    processed_at: { type: 'timestamp' }
  });
  pgm.createIndex('notes', 'user_id', { name: 'idx_notes_user_id' });
  pgm.createIndex('notes', 'subject_id', { name: 'idx_notes_subject_id' });
  pgm.createIndex('notes', 'chapter_id', { name: 'idx_notes_chapter_id' });
  pgm.createIndex('notes', 'processing_status', { name: 'idx_notes_processing_status' });
  pgm.createIndex('notes', 'created_at', { name: 'idx_notes_created_at' });
  pgm.createIndex('notes', 'parsed_content', { using: 'gin', name: 'idx_notes_parsed_content' });
  pgm.addConstraint('notes', 'notes_user_fk', { foreignKeys: [{ columns: 'user_id', references: 'users(id)', onDelete: 'cascade' }] });
  pgm.addConstraint('notes', 'notes_subject_fk', { foreignKeys: [{ columns: 'subject_id', references: 'subjects(id)', onDelete: 'set null' }] });
  pgm.addConstraint('notes', 'notes_chapter_fk', { foreignKeys: [{ columns: 'chapter_id', references: 'chapters(id)', onDelete: 'set null' }] });

  // flashcards
  pgm.createTable('flashcards', {
    id: { type: 'uuid', primaryKey: true, default: pgm.func('uuid_generate_v4()') },
    note_id: { type: 'uuid', notNull: true },
    user_id: { type: 'uuid', notNull: true },
    question: { type: 'text', notNull: true },
    answer: { type: 'text', notNull: true },
    difficulty: { type: 'integer', notNull: true, default: 0 },
    interval: { type: 'integer', notNull: true, default: 0 },
    repetitions: { type: 'integer', notNull: true, default: 0 },
    next_review_date: { type: 'date', notNull: true, default: pgm.func('CURRENT_DATE') },
    last_reviewed_at: { type: 'timestamp' },
    created_at: { type: 'timestamp', notNull: true, default: pgm.func('CURRENT_TIMESTAMP') },
    updated_at: { type: 'timestamp', notNull: true, default: pgm.func('CURRENT_TIMESTAMP') }
  });
  pgm.createIndex('flashcards', 'user_id', { name: 'idx_flashcards_user_id' });
  pgm.createIndex('flashcards', 'note_id', { name: 'idx_flashcards_note_id' });
  pgm.createIndex('flashcards', 'next_review_date', { name: 'idx_flashcards_next_review_date' });
  pgm.createIndex('flashcards', ['user_id', 'next_review_date'], { name: 'idx_flashcards_user_next_review' });
  pgm.addConstraint('flashcards', 'flashcards_note_fk', { foreignKeys: [{ columns: 'note_id', references: 'notes(id)', onDelete: 'cascade' }] });
  pgm.addConstraint('flashcards', 'flashcards_user_fk', { foreignKeys: [{ columns: 'user_id', references: 'users(id)', onDelete: 'cascade' }] });

  // study_sessions
  pgm.createTable('study_sessions', {
    id: { type: 'uuid', primaryKey: true, default: pgm.func('uuid_generate_v4()') },
    user_id: { type: 'uuid', notNull: true },
    flashcard_id: { type: 'uuid', notNull: true },
    quality: { type: 'integer', notNull: true },
    time_spent_seconds: { type: 'integer' },
    reviewed_at: { type: 'timestamp', notNull: true, default: pgm.func('CURRENT_TIMESTAMP') }
  });
  pgm.createIndex('study_sessions', 'user_id', { name: 'idx_sessions_user_id' });
  pgm.createIndex('study_sessions', 'flashcard_id', { name: 'idx_sessions_flashcard_id' });
  pgm.createIndex('study_sessions', 'reviewed_at', { name: 'idx_sessions_reviewed_at' });
  pgm.addConstraint('study_sessions', 'sessions_user_fk', { foreignKeys: [{ columns: 'user_id', references: 'users(id)', onDelete: 'cascade' }] });
  pgm.addConstraint('study_sessions', 'sessions_flashcard_fk', { foreignKeys: [{ columns: 'flashcard_id', references: 'flashcards(id)', onDelete: 'cascade' }] });

  // knowledge_graph_nodes
  pgm.createTable('knowledge_graph_nodes', {
    id: { type: 'uuid', primaryKey: true, default: pgm.func('uuid_generate_v4()') },
    user_id: { type: 'uuid', notNull: true },
    note_id: { type: 'uuid' },
    node_type: { type: 'varchar(50)', notNull: true },
    node_label: { type: 'varchar(255)', notNull: true },
    neptune_vertex_id: { type: 'varchar(255)' },
    properties: { type: 'jsonb' },
    created_at: { type: 'timestamp', notNull: true, default: pgm.func('CURRENT_TIMESTAMP') },
    updated_at: { type: 'timestamp', notNull: true, default: pgm.func('CURRENT_TIMESTAMP') }
  });
  pgm.createIndex('knowledge_graph_nodes', 'user_id', { name: 'idx_kg_user_id' });
  pgm.createIndex('knowledge_graph_nodes', 'note_id', { name: 'idx_kg_note_id' });
  pgm.createIndex('knowledge_graph_nodes', 'node_type', { name: 'idx_kg_node_type' });
  pgm.createIndex('knowledge_graph_nodes', 'neptune_vertex_id', { name: 'idx_kg_neptune_vertex_id' });
  pgm.addConstraint('knowledge_graph_nodes', 'kg_user_fk', { foreignKeys: [{ columns: 'user_id', references: 'users(id)', onDelete: 'cascade' }] });
  pgm.addConstraint('knowledge_graph_nodes', 'kg_note_fk', { foreignKeys: [{ columns: 'note_id', references: 'notes(id)', onDelete: 'cascade' }] });

  // refresh_tokens
  pgm.createTable('refresh_tokens', {
    id: { type: 'uuid', primaryKey: true, default: pgm.func('uuid_generate_v4()') },
    user_id: { type: 'uuid', notNull: true },
    token_hash: { type: 'varchar(255)', notNull: true },
    expires_at: { type: 'timestamp', notNull: true },
    created_at: { type: 'timestamp', notNull: true, default: pgm.func('CURRENT_TIMESTAMP') },
    revoked: { type: 'boolean', notNull: true, default: false }
  });
  pgm.createIndex('refresh_tokens', 'user_id', { name: 'idx_refresh_user_id' });
  pgm.createIndex('refresh_tokens', 'token_hash', { name: 'idx_refresh_token_hash' });
  pgm.createIndex('refresh_tokens', 'expires_at', { name: 'idx_refresh_expires_at' });
  pgm.addConstraint('refresh_tokens', 'refresh_user_fk', { foreignKeys: [{ columns: 'user_id', references: 'users(id)', onDelete: 'cascade' }] });
};

exports.down = (pgm) => {
  pgm.dropTable('refresh_tokens');
  pgm.dropTable('knowledge_graph_nodes');
  pgm.dropTable('study_sessions');
  pgm.dropTable('flashcards');
  pgm.dropTable('notes');
  pgm.dropTable('chapters');
  pgm.dropTable('subjects');
  pgm.dropTable('users');
  pgm.dropExtension('uuid-ossp');
};

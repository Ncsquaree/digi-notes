exports.up = (pgm) => {
  pgm.sql(`
    CREATE OR REPLACE FUNCTION trigger_set_timestamp()
    RETURNS TRIGGER AS $$
    BEGIN
      NEW.updated_at = CURRENT_TIMESTAMP;
      RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
  `);

  const tables = ['users','subjects','chapters','notes','flashcards','knowledge_graph_nodes'];
  tables.forEach((t) => {
    pgm.sql(`CREATE TRIGGER set_timestamp BEFORE UPDATE ON ${t} FOR EACH ROW EXECUTE FUNCTION trigger_set_timestamp();`);
  });
};

exports.down = (pgm) => {
  const tables = ['users','subjects','chapters','notes','flashcards','knowledge_graph_nodes'];
  tables.forEach((t) => {
    pgm.sql(`DROP TRIGGER IF EXISTS set_timestamp ON ${t};`);
  });
  pgm.sql('DROP FUNCTION IF EXISTS trigger_set_timestamp();');
};

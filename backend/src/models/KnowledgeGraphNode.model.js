const db = require('../config/database');

class KnowledgeGraphNode {
  static async findById(nodeId) {
    const res = await db.pool.query('SELECT * FROM knowledge_graph_nodes WHERE id = $1 LIMIT 1', [nodeId]);
    return res.rowCount ? res.rows[0] : null;
  }

  static async findByUserId(userId, { limit = 100, offset = 0, filters = {} } = {}) {
    const clauses = ['user_id = $1'];
    const values = [userId];
    let idx = 2;
    if (filters.node_type) { clauses.push(`node_type = $${idx++}`); values.push(filters.node_type); }
    if (filters.note_id) { clauses.push(`note_id = $${idx++}`); values.push(filters.note_id); }
    const where = `WHERE ${clauses.join(' AND ')}`;
    const sql = `SELECT * FROM knowledge_graph_nodes ${where} ORDER BY created_at DESC LIMIT $${idx++} OFFSET $${idx}`;
    values.push(limit, offset);
    const res = await db.pool.query(sql, values);
    return res.rows;
  }

  static async findByNoteId(noteId) {
    const res = await db.pool.query('SELECT * FROM knowledge_graph_nodes WHERE note_id = $1 ORDER BY created_at DESC', [noteId]);
    return res.rows;
  }

  static async findByNeptuneVertexId(neptuneVertexId) {
    const res = await db.pool.query('SELECT * FROM knowledge_graph_nodes WHERE neptune_vertex_id = $1 LIMIT 1', [neptuneVertexId]);
    return res.rowCount ? res.rows[0] : null;
  }

  static async findByNodeType(userId, nodeType, { limit = 100 } = {}) {
    const res = await db.pool.query('SELECT * FROM knowledge_graph_nodes WHERE user_id = $1 AND node_type = $2 ORDER BY created_at DESC LIMIT $3', [userId, nodeType, limit]);
    return res.rows;
  }

  static async create({ user_id, note_id = null, node_type, node_label, neptune_vertex_id = null, properties = {} }) {
    const res = await db.pool.query(
      `INSERT INTO knowledge_graph_nodes (user_id, note_id, node_type, node_label, neptune_vertex_id, properties) VALUES ($1,$2,$3,$4,$5,$6) RETURNING *`,
      [user_id, note_id, node_type, node_label, neptune_vertex_id, properties]
    );
    return res.rows[0];
  }

  static async update(nodeId, fields = {}) {
    const keys = Object.keys(fields);
    if (!keys.length) return this.findById(nodeId);

    const allowed = ['note_id', 'node_type', 'node_label', 'neptune_vertex_id', 'properties'];
    const validKeys = keys.filter(k => allowed.includes(k));
    if (!validKeys.length) return this.findById(nodeId);

    const sets = validKeys.map((k, i) => `${k} = $${i + 2}`);
    const values = [nodeId, ...validKeys.map((k) => fields[k])];
    const sql = `UPDATE knowledge_graph_nodes SET ${sets.join(', ')} WHERE id = $1 RETURNING *`;
    const res = await db.pool.query(sql, values);
    return res.rowCount ? res.rows[0] : null;
  }

  static async delete(nodeId) {
    await db.pool.query('DELETE FROM knowledge_graph_nodes WHERE id = $1', [nodeId]);
    return true;
  }

  static async bulkCreate(nodesArray = []) {
    if (!nodesArray.length) return [];
    const cols = ['user_id','note_id','node_type','node_label','neptune_vertex_id','properties'];
    const values = [];
    const placeholders = nodesArray.map((n, i) => {
      const idx = i * cols.length;
      values.push(n.user_id, n.note_id || null, n.node_type, n.node_label, n.neptune_vertex_id || null, n.properties || {});
      return `($${idx+1},$${idx+2},$${idx+3},$${idx+4},$${idx+5},$${idx+6})`;
    }).join(',');
    const sql = `INSERT INTO knowledge_graph_nodes (${cols.join(',')}) VALUES ${placeholders} RETURNING *`;
    const res = await db.pool.query(sql, values);
    return res.rows;
  }

  static async countByUserId(userId, filters = {}) {
    const clauses = ['user_id = $1'];
    const values = [userId];
    let idx = 2;
    if (filters.node_type) { clauses.push(`node_type = $${idx++}`); values.push(filters.node_type); }
    const where = `WHERE ${clauses.join(' AND ')}`;
    const sql = `SELECT COUNT(*)::int as cnt FROM knowledge_graph_nodes ${where}`;
    const res = await db.pool.query(sql, values);
    return res.rows[0].cnt;
  }
}

module.exports = KnowledgeGraphNode;

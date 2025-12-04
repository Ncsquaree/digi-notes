// Mock database pool for tests
const mockClient = () => {
  const client = {
    query: jest.fn(async (sql, params) => ({ rowCount: 0, rows: [] })),
    release: jest.fn(),
  };
  return client;
};

const mockPool = {
  query: jest.fn(async (sql, params) => ({ rowCount: 0, rows: [] })),
  connect: jest.fn(async () => {
    const client = mockClient();
    client.query = jest.fn(async (sql, params) => {
      if (sql && sql.toUpperCase && sql.toUpperCase().includes('BEGIN')) return { rows: [] };
      return { rowCount: 0, rows: [] };
    });
    client.release = jest.fn();
    return client;
  }),
};

function resetMockPool() {
  mockPool.query.mockReset && mockPool.query.mockReset();
  mockPool.connect.mockReset && mockPool.connect.mockReset();
}

module.exports = { pool: mockPool, mockPool, mockClient, resetMockPool };

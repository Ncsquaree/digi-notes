const fs = require('fs');
const path = require('path');
try {
  const { swaggerSpec } = require('../src/config/swagger');
  const out = path.resolve(__dirname, '..', 'api-docs.json');
  fs.writeFileSync(out, JSON.stringify(swaggerSpec, null, 2));
  console.log('Wrote swagger spec to', out);
} catch (e) {
  console.error('Failed to generate swagger spec', e.message || e);
  process.exit(1);
}

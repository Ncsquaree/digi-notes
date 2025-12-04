#!/bin/sh
set -e

echo "Running environment validation..."
node scripts/validate-env.js

echo "Validation succeeded. Starting server..."
exec node server.js

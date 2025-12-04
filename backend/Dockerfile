FROM node:18-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json* ./
# Install production dependencies; fail build on error
RUN npm ci --only=production
# Install dev dependencies if present; fail build on error
RUN npm ci

FROM node:18-alpine
RUN apk add --no-cache curl
RUN addgroup -S nodegroup && adduser -S nodeuser -G nodegroup -u 1000
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . /app
RUN chown -R nodeuser:nodegroup /app
ENV NODE_ENV=production
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:5000/health || exit 1
# Ensure the startup wrapper is executable and run it as the container CMD
RUN chmod +x /app/validate-and-start.sh || true
USER nodeuser
CMD ["sh","/app/validate-and-start.sh"]

FROM node:20-bookworm-slim AS builder

RUN apt-get update && apt-get install -y \
    python3 make g++ git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json package-lock.json ./
COPY packages/highlight/package.json packages/highlight/
COPY packages/relay/package.json packages/relay/
COPY packages/server/package.json packages/server/
COPY packages/cli/package.json packages/cli/

RUN npm ci --ignore-scripts

COPY . .

RUN npm rebuild node-pty onnxruntime-node
RUN npm run build:daemon

FROM node:20-bookworm-slim

RUN apt-get update && apt-get install -y \
    tini bzip2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/package.json /app/package-lock.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/packages/highlight ./packages/highlight
COPY --from=builder /app/packages/relay ./packages/relay
COPY --from=builder /app/packages/server ./packages/server
COPY --from=builder /app/packages/cli ./packages/cli

ENV NODE_ENV=production
ENV PASEO_LISTEN=0.0.0.0:6767

EXPOSE 6767

ENTRYPOINT ["tini", "--"]
CMD ["npm", "start"]

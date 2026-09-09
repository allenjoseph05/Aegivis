# syntax=docker/dockerfile:1
# To pin base images for reproducible production builds, retrieve digests with:
#   docker pull node:20-alpine && docker inspect node:20-alpine --format '{{index .RepoDigests 0}}'
#   docker pull nginx:alpine   && docker inspect nginx:alpine   --format '{{index .RepoDigests 0}}'
# Then replace FROM lines below with: FROM <image>@sha256:<digest>
FROM node:20-alpine AS builder

WORKDIR /app

# Copy lock file first so npm install layer is cached unless deps change
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline

COPY . .

# Edition is baked into the bundle at build time.
# Pass --build-arg VITE_EDITION=enterprise for the enterprise image.
ARG VITE_EDITION=community
ENV VITE_EDITION=$VITE_EDITION

RUN npm run build

# Serve with nginx
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

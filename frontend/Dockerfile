FROM node:24-slim

WORKDIR /app

COPY docker-entrypoint.sh /usr/local/bin/frontend-entrypoint.sh
RUN chmod +x /usr/local/bin/frontend-entrypoint.sh

EXPOSE 3000

ENTRYPOINT ["/usr/local/bin/frontend-entrypoint.sh"]

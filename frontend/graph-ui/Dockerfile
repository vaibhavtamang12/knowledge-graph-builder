FROM node:18-alpine

WORKDIR /app

COPY frontend/graph-ui/package*.json ./
RUN npm install

COPY frontend/graph-ui/ ./

EXPOSE 3000
CMD ["npm", "start"]
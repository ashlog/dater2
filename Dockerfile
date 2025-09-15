# Use the official Node.js image as a base
FROM node:18

# Set the working directory
WORKDIR /app

COPY . .
# Install dependencies
RUN cd functions && npm install

# Expose the port the app runs on (adjust if necessary)
EXPOSE 3000

WORKDIR /app/functions
# Start the application
CMD ["npm", "run", "dev"]
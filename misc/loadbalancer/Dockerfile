# Start from the official golang base image
FROM golang:latest

# Set the working directory inside the container
WORKDIR /app

# Copy go.mod and go.sum files to the working directory
COPY go.mod go.sum ./

# Download dependencies if any
RUN go mod download

# Copy the rest of the application source code to the working directory
COPY . .

# Build the Go application
RUN go build -o main .

# Expose port 4000 for the API
EXPOSE 4000

# Set environment variables
ARG ALFA_BASE_URL=""
ARG BETA_BASE_URL=""

ENV ALFA_BASE_URL="${ALFA_BASE_URL}"
ENV BETA_BASE_URL="${BETA_BASE_URL}"

# Command to run the application
CMD ["./main"]

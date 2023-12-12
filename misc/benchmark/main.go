package main

import (
	"fmt"
	"net/http"
)

func main() {
	// Define a handler function for incoming requests
	handler := func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Hello, this is a basic HTTP server in Go!")
	}

	// Register the handler function for a specific route pattern
	http.HandleFunc("/", handler)

	// Start the HTTP server on port 8080
	fmt.Println("Server listening on port 8080")
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		fmt.Println("Error:", err)
	}
}

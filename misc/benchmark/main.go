package main

import (
	"fmt"
	"net/http"
	"time"
)

func main() {
	// Create a channel to receive output
	outputChan := make(chan string)

	// Define a handler function for incoming requests
	handler := func(w http.ResponseWriter, r *http.Request) {
		// Log the start time
		startTime := time.Now()
		startFormatted := startTime.Format("15:04:05")
		fmt.Println("Request received at:", startFormatted)

		// Start a goroutine to handle the task
		go func() {
			// Simulate a time-consuming task with a 3-second delay
			time.Sleep(3 * time.Second)

			// Send the response after the delay
			response := "Hello, this is a basic HTTP server in Go! After a 3-second delay"
			outputChan <- response
		}()
	}

	// Register the handler function for a specific route pattern
	http.HandleFunc("/", handler)

	// Start the HTTP server on port 8080
	fmt.Println("Server listening on port 8000")
	err := http.ListenAndServe(":8000", nil)
	if err != nil {
		fmt.Println("Error:", err)
	}

	// Retrieve output from the channel
	go func() {
		for {
			select {
			case output := <-outputChan:
				// Handle the received output
				fmt.Println("Received output:", output)

				// You can add other cases or conditions as needed
			}
		}
	}()

	// Keep the main goroutine running
	select {}
}

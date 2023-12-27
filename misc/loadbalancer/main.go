package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"time"

	"github.com/gofiber/fiber/v2"
)

/*
	STRUCT
*/

type GPUResponse struct {
	Error    bool      `json:"error"`
	Response []GPUInfo `json:"response"`
}

type GPUInfo struct {
	Index             int    `json:"index"`
	UUID              string `json:"uuid"`
	Name              string `json:"name"`
	MemoryTotal       int    `json:"memory_total"`
	MemoryUsed        int    `json:"memory_used"`
	MemoryFree        int    `json:"memory_free"`
	UtilizationGPU    int    `json:"utilization_gpu"`
	UtilizationMemory int    `json:"utilization_memory"`
}

type RequestBody struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

/*
	Helper
*/

func makeImageProcessingRequest(text string, URL string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	// Prepare the request body
	requestBody := RequestBody{
		Type: "image",
		Text: text,
	}
	requestBodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return "", err
	}

	// Create HTTP client with timeout
	client := &http.Client{}
	req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s", URL), bytes.NewBuffer(requestBodyBytes))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	// Perform the HTTP request with timeout
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	// Read the response body
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	return string(body), nil
}

func makeTextProcessingRequest(text string, URL string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	// Prepare the request body
	requestBody := RequestBody{
		Type: "text",
		Text: text,
	}
	requestBodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return "", err
	}

	// Create HTTP client with timeout
	client := &http.Client{}
	req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s", URL), bytes.NewBuffer(requestBodyBytes))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	// Perform the HTTP request with timeout
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	// Read the response body
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	return string(body), nil
}

func checkGPU(url string) (GPUInfo, error) {
	var gpuInfo GPUInfo
	var gpuResp GPUResponse

	client := http.Client{
		Timeout: 25 * time.Second, // Timeout for the HTTP GET request
	}

	// Make GET request to the provided URL
	resp, err := client.Get(fmt.Sprintf("%s/check-gpu", url))
	if err != nil {
		return gpuInfo, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return gpuInfo, fmt.Errorf("GET request failed with status code: %d", resp.StatusCode)
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return gpuInfo, err
	}

	err = json.Unmarshal(body, &gpuResp)
	if err != nil {
		return gpuInfo, err
	}
	return gpuResp.Response[0], nil
}

/*
	MAIN
*/

func main() {
	app := fiber.New()

	// Handle POST requests to /test
	app.Post("/test", func(c *fiber.Ctx) error {
		var reqBody RequestBody
		if err := c.BodyParser(&reqBody); err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Invalid JSON",
			})
		}
		URL := ""

		alfaBaseURL := os.Getenv("ALFA_BASE_URL")
		betaBaseURL := os.Getenv("BETA_BASE_URL")

		if alfaBaseURL == "" || betaBaseURL == "" {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": "ALFA_BASE_URL or BETA_BASE_URL environment variable not set",
			})
		}

		alfa_gpu, err_a := checkGPU(alfaBaseURL)
		beta_gpu, err_b := checkGPU(betaBaseURL)

		if err_a == nil && err_b == nil {
			if alfa_gpu.MemoryUsed <= beta_gpu.MemoryUsed {
				URL += alfaBaseURL
			} else {
				URL += betaBaseURL
			}
		} else {
			if err_a != nil && err_b != nil {
				return c.Status(fiber.StatusInternalServerError).SendString("Error")
			} else {
				if err_a != nil {
					URL += betaBaseURL
				} else {
					URL += alfaBaseURL
				}
			}
		}

		res := ""
		var err error
		err = nil

		if reqBody.Type == "image" {
			URL += "/inference-image"
			res, err = makeImageProcessingRequest(reqBody.Text, URL)
			if err != nil {
				return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
					"error": "ALFA_BASE_URL or BETA_BASE_URL environment variable not set",
				})
			}
			res = "[Object]"
		} else {
			URL += "/inference-text"
			res, err = makeTextProcessingRequest(reqBody.Text, URL)
			if err != nil {
				return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
					"error": "ALFA_BASE_URL or BETA_BASE_URL environment variable not set",
				})
			}
			res = "[Text]"
		}

		return c.SendString(res)
	})

	// Handle POST requests to /generate
	app.Post("/generate", func(c *fiber.Ctx) error {
		var reqBody RequestBody
		if err := c.BodyParser(&reqBody); err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Invalid JSON",
			})
		}
		URL := ""

		alfaBaseURL := os.Getenv("ALFA_BASE_URL")
		betaBaseURL := os.Getenv("BETA_BASE_URL")

		if alfaBaseURL == "" || betaBaseURL == "" {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": "ALFA_BASE_URL or BETA_BASE_URL environment variable not set",
			})
		}

		alfa_gpu, err_a := checkGPU(alfaBaseURL)
		beta_gpu, err_b := checkGPU(betaBaseURL)

		if err_a == nil && err_b == nil {
			if alfa_gpu.MemoryUsed <= beta_gpu.MemoryUsed {
				URL += alfaBaseURL
			} else {
				URL += betaBaseURL
			}
		} else {
			if err_a != nil && err_b != nil {
				return c.Status(fiber.StatusInternalServerError).SendString("Error")
			} else {
				if err_a != nil {
					URL += betaBaseURL
				} else {
					URL += alfaBaseURL
				}
			}
		}

		res := ""
		var err error
		err = nil
		URL += "/generate"
		if reqBody.Type == "image" {
			res, err = makeImageProcessingRequest(reqBody.Text, URL)
			if err != nil {
				return c.Status(fiber.StatusInternalServerError).SendString("https://cdn.discordapp.com/attachments/1102768794629328929/1189397784076488774/Screenshot_2023-12-27_at_09.42.04.png?ex=659e0401&is=658b8f01&hm=dc28fe9e9a85a8bbdf40ff800af220d3984fca949ffb0c706e784e732faad482")
			}
		} else {
			res, err = makeTextProcessingRequest(reqBody.Text, URL)
			if err != nil {
				return c.Status(fiber.StatusInternalServerError).SendString("Error&")
			}
		}

		return c.SendString(res)
	})

	app.Listen(":4000")
}

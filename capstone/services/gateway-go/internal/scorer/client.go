// Package scorer memanggil Python inference service (POST /v1/classify).
package scorer

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/ariskaAdi/ml-bootcamp/capstone/services/gateway-go/internal/filter"
)

var (
	ErrTimeout     = errors.New("inference timeout")
	ErrUnavailable = errors.New("inference unavailable")
	ErrBadResponse = errors.New("inference bad response")
)

type RequestIDKey struct{}

type Client struct {
	BaseURL string
	HTTP    *http.Client
	Timeout time.Duration
}

func New(baseURL string, timeout time.Duration) *Client {
	return &Client{BaseURL: baseURL, HTTP: &http.Client{}, Timeout: timeout}
}

type classifyResponse struct {
	ModelVersion string `json:"model_version"`
	Predictions  []struct {
		SpamProbability float64 `json:"spam_probability"`
		IsSpam          bool    `json:"is_spam"`
	} `json:"predictions"`
}

func (c *Client) Score(ctx context.Context, text string) (filter.Score, error) {
	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()
	body, _ := json.Marshal(map[string]any{"instances": []map[string]string{{"text": text}}})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/v1/classify", bytes.NewReader(body))
	if err != nil {
		return filter.Score{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if rid, ok := ctx.Value(RequestIDKey{}).(string); ok {
		req.Header.Set("X-Request-ID", rid)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return filter.Score{}, ErrTimeout
		}
		return filter.Score{}, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 200))
		return filter.Score{}, fmt.Errorf("%w: status %d: %s", ErrBadResponse, resp.StatusCode, snippet)
	}
	var cr classifyResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil || len(cr.Predictions) != 1 {
		return filter.Score{}, fmt.Errorf("%w: decode", ErrBadResponse)
	}
	p := cr.Predictions[0]
	return filter.Score{Probability: p.SpamProbability, IsSpam: p.IsSpam, ModelVersion: cr.ModelVersion}, nil
}

func (c *Client) Ready(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/ready", nil)
	if err != nil {
		return err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%w: status %d", ErrUnavailable, resp.StatusCode)
	}
	return nil
}

// Package inference memanggil Python inference service lewat HTTP/JSON internal.
package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/ariskaAdi/ml-bootcamp/services/api-go/internal/predict"
)

var (
	ErrTimeout     = errors.New("inference timeout")
	ErrUnavailable = errors.New("inference service unavailable")
	ErrBadResponse = errors.New("inference service returned an error")
)

// RequestIDKey adalah kunci context untuk meneruskan request id dari Go ke Python.
type RequestIDKey struct{}

type Client struct {
	BaseURL string
	HTTP    *http.Client
	Timeout time.Duration // batas waktu PER request ke Python
}

func New(baseURL string, timeout time.Duration) *Client {
	return &Client{BaseURL: baseURL, HTTP: &http.Client{}, Timeout: timeout}
}

type predictRequest struct {
	Instances []predict.Features `json:"instances"`
}

type predictResponse struct {
	ModelVersion string  `json:"model_version"`
	Threshold    float64 `json:"threshold"`
	Predictions  []struct {
		Probability float64 `json:"probability"`
		ShouldCall  bool    `json:"should_call"`
	} `json:"predictions"`
}

// Score memenuhi interface predict.Scorer.
func (c *Client) Score(ctx context.Context, f predict.Features) (predict.Score, error) {
	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()

	body, err := json.Marshal(predictRequest{Instances: []predict.Features{f}})
	if err != nil {
		return predict.Score{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/v1/predict", bytes.NewReader(body))
	if err != nil {
		return predict.Score{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if rid, ok := ctx.Value(RequestIDKey{}).(string); ok {
		req.Header.Set("X-Request-ID", rid) // request id yang sama di log Go dan Python
	}

	resp, err := c.HTTP.Do(req)
	if err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return predict.Score{}, ErrTimeout
		}
		return predict.Score{}, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 200))
		return predict.Score{}, fmt.Errorf("%w: status %d: %s", ErrBadResponse, resp.StatusCode, snippet)
	}
	var pr predictResponse
	if err := json.NewDecoder(resp.Body).Decode(&pr); err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return predict.Score{}, ErrTimeout
		}
		return predict.Score{}, fmt.Errorf("%w: decode: %v", ErrBadResponse, err)
	}
	if len(pr.Predictions) != 1 {
		return predict.Score{}, fmt.Errorf("%w: expected 1 prediction, got %d", ErrBadResponse, len(pr.Predictions))
	}
	p := pr.Predictions[0]
	return predict.Score{Probability: p.Probability, ModelCall: p.ShouldCall, Threshold: pr.Threshold,
		ModelVersion: pr.ModelVersion}, nil
}

// Ready mengecek /ready service Python (dipakai endpoint /ready Go).
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

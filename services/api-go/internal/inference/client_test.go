package inference

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ariskaAdi/ml-bootcamp/services/api-go/internal/predict"
)

// httptest.Server = server HTTP sungguhan di memori, untuk meniru service Python.
func fakePython(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return New(srv.URL, 200*time.Millisecond)
}

func TestScoreSuccessAndForwardsRequestID(t *testing.T) {
	var gotRID string
	var gotBody predictRequest
	c := fakePython(t, func(w http.ResponseWriter, r *http.Request) {
		gotRID = r.Header.Get("X-Request-ID")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{"model_version":"v1","threshold":0.12,"predictions":[{"probability":0.3,"should_call":true}]}`))
	})
	ctx := context.WithValue(context.Background(), RequestIDKey{}, "rid-1")
	s, err := c.Score(ctx, predict.Features{Age: 40})
	if err != nil {
		t.Fatal(err)
	}
	if s.Probability != 0.3 || !s.ModelCall || s.ModelVersion != "v1" {
		t.Fatalf("unexpected score %+v", s)
	}
	if gotRID != "rid-1" || len(gotBody.Instances) != 1 || gotBody.Instances[0].Age != 40 {
		t.Fatalf("request not forwarded correctly: rid=%q body=%+v", gotRID, gotBody)
	}
}

func TestScoreTimeout(t *testing.T) {
	c := fakePython(t, func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(2 * time.Second): // lebih lama dari timeout klien (200ms)
		case <-r.Context().Done():
		}
	})
	start := time.Now()
	_, err := c.Score(context.Background(), predict.Features{})
	if !errors.Is(err, ErrTimeout) {
		t.Fatalf("expected ErrTimeout, got %v", err)
	}
	if time.Since(start) > time.Second {
		t.Fatal("client did not respect timeout")
	}
}

func TestScoreUpstream500(t *testing.T) {
	c := fakePython(t, func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"detail":"inference failed"}`, http.StatusInternalServerError)
	})
	if _, err := c.Score(context.Background(), predict.Features{}); !errors.Is(err, ErrBadResponse) {
		t.Fatalf("expected ErrBadResponse, got %v", err)
	}
}

func TestScoreMalformedJSON(t *testing.T) {
	c := fakePython(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{not json`)) })
	if _, err := c.Score(context.Background(), predict.Features{}); !errors.Is(err, ErrBadResponse) {
		t.Fatalf("expected ErrBadResponse, got %v", err)
	}
}

func TestScoreServiceDown(t *testing.T) {
	c := New("http://127.0.0.1:1", 200*time.Millisecond) // tidak ada yang listen
	if _, err := c.Score(context.Background(), predict.Features{}); !errors.Is(err, ErrUnavailable) && !errors.Is(err, ErrTimeout) {
		t.Fatalf("expected ErrUnavailable, got %v", err)
	}
}

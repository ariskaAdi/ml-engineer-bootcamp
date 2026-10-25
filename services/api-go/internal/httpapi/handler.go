// Package httpapi berisi handler HTTP: decode JSON, panggil service, petakan error ke status code.
// Tidak ada logika bisnis di sini.
package httpapi

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"time"

	"github.com/ariskaAdi/ml-bootcamp/services/api-go/internal/inference"
	"github.com/ariskaAdi/ml-bootcamp/services/api-go/internal/predict"
)

const maxBodyBytes = 1 << 16 // 64 KB cukup untuk satu nasabah

type Decider interface {
	Decide(ctx context.Context, r predict.ClientRequest) (predict.Decision, error)
}

type ReadyChecker interface {
	Ready(ctx context.Context) error
}

type Handler struct {
	Service Decider
	Ready   ReadyChecker
	Log     *slog.Logger
}

func (h *Handler) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", h.health)
	mux.HandleFunc("GET /ready", h.ready)
	mux.HandleFunc("POST /v1/predictions", h.predict)
	return h.withRequestLog(mux)
}

func (h *Handler) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (h *Handler) ready(w http.ResponseWriter, r *http.Request) {
	if err := h.Ready.Ready(r.Context()); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "not ready"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func (h *Handler) predict(w http.ResponseWriter, r *http.Request) {
	var req predict.ClientRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	dec.DisallowUnknownFields() // typo nama field -> error, bukan diabaikan diam-diam
	if err := dec.Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "malformed JSON", []string{err.Error()})
		return
	}
	decision, err := h.Service.Decide(r.Context(), req)
	var verr *predict.ValidationError
	switch {
	case err == nil:
		writeJSON(w, http.StatusOK, decision)
	case errors.As(err, &verr):
		writeError(w, http.StatusUnprocessableEntity, "validation failed", verr.Problems)
	case errors.Is(err, inference.ErrTimeout):
		h.Log.Warn("inference timeout", "request_id", requestID(r.Context()))
		writeError(w, http.StatusGatewayTimeout, "model service timeout", nil)
	case errors.Is(err, inference.ErrUnavailable):
		h.Log.Error("inference unavailable", "request_id", requestID(r.Context()), "error", err.Error())
		writeError(w, http.StatusServiceUnavailable, "model service unavailable", nil)
	default:
		h.Log.Error("inference failed", "request_id", requestID(r.Context()), "error", err.Error())
		writeError(w, http.StatusBadGateway, "model service error", nil)
	}
}

// withRequestLog: request id + satu baris log terstruktur per request. Isi body TIDAK dicatat.
func (h *Handler) withRequestLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rid := r.Header.Get("X-Request-ID")
		if rid == "" {
			rid = newID()
		}
		ctx := context.WithValue(r.Context(), inference.RequestIDKey{}, rid)
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		rec.Header().Set("X-Request-ID", rid)
		start := time.Now()
		next.ServeHTTP(rec, r.WithContext(ctx))
		h.Log.Info("request", "request_id", rid, "method", r.Method, "path", r.URL.Path,
			"status", rec.status, "latency_ms", float64(time.Since(start).Microseconds())/1000)
	})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (s *statusRecorder) WriteHeader(code int) {
	s.status = code
	s.ResponseWriter.WriteHeader(code)
}

func requestID(ctx context.Context) string {
	v, _ := ctx.Value(inference.RequestIDKey{}).(string)
	return v
}

func newID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string, details []string) {
	body := map[string]any{"error": msg}
	if len(details) > 0 {
		body["details"] = details
	}
	writeJSON(w, status, body)
}

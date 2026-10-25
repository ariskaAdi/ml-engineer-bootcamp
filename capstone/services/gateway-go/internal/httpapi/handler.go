// Package httpapi: decode JSON, panggil filter.Service, petakan error ke status code.
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

	"github.com/ariskaAdi/ml-bootcamp/capstone/services/gateway-go/internal/filter"
	"github.com/ariskaAdi/ml-bootcamp/capstone/services/gateway-go/internal/scorer"
)

const maxBodyBytes = 16 << 10

type Classifier interface {
	Classify(ctx context.Context, m filter.Message) (filter.Verdict, error)
}

type ReadyChecker interface {
	Ready(ctx context.Context) error
}

type Handler struct {
	Service Classifier
	Ready   ReadyChecker
	Log     *slog.Logger
}

func (h *Handler) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /ready", func(w http.ResponseWriter, r *http.Request) {
		if err := h.Ready.Ready(r.Context()); err != nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "not ready"})
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
	})
	mux.HandleFunc("POST /v1/messages/classify", h.classify)
	return h.withRequestLog(mux)
}

func (h *Handler) classify(w http.ResponseWriter, r *http.Request) {
	var m filter.Message
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		writeError(w, http.StatusBadRequest, "malformed JSON", []string{err.Error()})
		return
	}
	v, err := h.Service.Classify(r.Context(), m)
	var verr *filter.ValidationError
	switch {
	case err == nil:
		if v.Degraded {
			h.Log.Warn("degraded verdict", "request_id", rid(r.Context()), "message_id", m.MessageID)
		}
		writeJSON(w, http.StatusOK, v)
	case errors.As(err, &verr):
		writeError(w, http.StatusUnprocessableEntity, "validation failed", verr.Problems)
	case errors.Is(err, scorer.ErrTimeout):
		writeError(w, http.StatusGatewayTimeout, "model service timeout", nil)
	case errors.Is(err, scorer.ErrUnavailable):
		writeError(w, http.StatusServiceUnavailable, "model service unavailable", nil)
	default:
		h.Log.Error("classify failed", "request_id", rid(r.Context()), "error", err.Error())
		writeError(w, http.StatusBadGateway, "model service error", nil)
	}
}

// Log per request: TANPA isi pesan dan TANPA nomor pengirim (data pribadi).
func (h *Handler) withRequestLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" {
			b := make([]byte, 16)
			_, _ = rand.Read(b)
			id = hex.EncodeToString(b)
		}
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		rec.Header().Set("X-Request-ID", id)
		start := time.Now()
		next.ServeHTTP(rec, r.WithContext(context.WithValue(r.Context(), scorer.RequestIDKey{}, id)))
		h.Log.Info("request", "request_id", id, "method", r.Method, "path", r.URL.Path, "status", rec.status,
			"latency_ms", float64(time.Since(start).Microseconds())/1000)
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

func rid(ctx context.Context) string {
	v, _ := ctx.Value(scorer.RequestIDKey{}).(string)
	return v
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

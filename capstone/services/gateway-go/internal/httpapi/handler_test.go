package httpapi

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ariskaAdi/ml-bootcamp/capstone/services/gateway-go/internal/filter"
	"github.com/ariskaAdi/ml-bootcamp/capstone/services/gateway-go/internal/scorer"
)

// Tes integrasi kecil: handler + filter.Service + scorer.Client asli, dengan Python palsu (httptest).
func newStack(t *testing.T, python http.HandlerFunc, failOpen bool) http.Handler {
	t.Helper()
	py := httptest.NewServer(python)
	t.Cleanup(py.Close)
	c := scorer.New(py.URL, 200*time.Millisecond)
	h := &Handler{Service: &filter.Service{Scorer: c, FailOpen: failOpen}, Ready: c,
		Log: slog.New(slog.NewJSONHandler(io.Discard, nil))}
	return h.Routes()
}

func pythonReturning(p float64, isSpam bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Instances []struct{ Text string } `json:"instances"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if len(body.Instances) != 1 || body.Instances[0].Text == "" {
			http.Error(w, "bad", 422)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"model_version": "v-test", "threshold": 0.43,
			"predictions": []map[string]any{{"spam_probability": p, "is_spam": isSpam}}})
	}
}

func post(h http.Handler, body string) (*httptest.ResponseRecorder, map[string]any) {
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages/classify", strings.NewReader(body)))
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec, out
}

const okBody = `{"message_id":"m1","sender":"+62811","sender_is_contact":false,"text":"WIN a FREE prize now"}`

func TestEndToEndQuarantine(t *testing.T) {
	rec, out := post(newStack(t, pythonReturning(0.98, true), true), okBody)
	if rec.Code != 200 || out["action"] != "quarantine" || out["model_version"] != "v-test" {
		t.Fatalf("got %d %v", rec.Code, out)
	}
}

func TestPythonTimeoutFailOpenAndFailClosed(t *testing.T) {
	slow := func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(2 * time.Second):
		case <-r.Context().Done():
		}
	}
	rec, out := post(newStack(t, slow, true), okBody)
	if rec.Code != 200 || out["action"] != "deliver" || out["degraded"] != true {
		t.Fatalf("fail-open: got %d %v", rec.Code, out)
	}
	rec, _ = post(newStack(t, slow, false), okBody)
	if rec.Code != http.StatusGatewayTimeout {
		t.Fatalf("fail-closed: got %d", rec.Code)
	}
}

func TestBadRequests(t *testing.T) {
	h := newStack(t, pythonReturning(0.1, false), true)
	for body, want := range map[string]int{
		`{not json`:                       400,
		`{"message_id":"m","txt":"typo"}`: 400,
		`{"message_id":"","text":""}`:     422,
	} {
		if rec, _ := post(h, body); rec.Code != want {
			t.Errorf("%s: got %d want %d", body, rec.Code, want)
		}
	}
}

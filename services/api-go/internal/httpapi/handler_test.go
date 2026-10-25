package httpapi

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ariskaAdi/ml-bootcamp/services/api-go/internal/inference"
	"github.com/ariskaAdi/ml-bootcamp/services/api-go/internal/predict"
)

type fakeDecider struct {
	d   predict.Decision
	err error
}

func (f fakeDecider) Decide(_ context.Context, r predict.ClientRequest) (predict.Decision, error) {
	f.d.ClientID = r.ClientID
	return f.d, f.err
}

type fakeReady struct{ err error }

func (f fakeReady) Ready(context.Context) error { return f.err }

func newServer(d fakeDecider) http.Handler {
	h := &Handler{Service: d, Ready: fakeReady{}, Log: slog.New(slog.NewJSONHandler(io.Discard, nil))}
	return h.Routes()
}

const body = `{"client_id":"C1","age":35,"job":"admin.","marital":"married","education":"university.degree",
"default":"no","housing":"yes","loan":"no","contact":"cellular","month":"may","day_of_week":"thu","campaign":1,
"pdays":999,"previous":0,"poutcome":"nonexistent","emp_var_rate":-1.8,"cons_price_idx":92.893,
"cons_conf_idx":-46.2,"euribor3m":1.313,"nr_employed":5099.1}`

func do(h http.Handler, method, path, payload string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(method, path, strings.NewReader(payload)))
	return rec
}

func TestPredictOK(t *testing.T) {
	h := newServer(fakeDecider{d: predict.Decision{Action: "call", Priority: "high", Probability: 0.7}})
	rec := do(h, http.MethodPost, "/v1/predictions", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var d predict.Decision
	_ = json.Unmarshal(rec.Body.Bytes(), &d)
	if d.ClientID != "C1" || d.Action != "call" {
		t.Fatalf("unexpected body %+v", d)
	}
	if rec.Header().Get("X-Request-ID") == "" {
		t.Fatal("missing X-Request-ID")
	}
}

func TestPredictErrorMapping(t *testing.T) {
	cases := []struct {
		err  error
		want int
	}{
		{&predict.ValidationError{Problems: []string{"age"}}, http.StatusUnprocessableEntity},
		{fmt.Errorf("%w: %w", predict.ErrScoring, inference.ErrTimeout), http.StatusGatewayTimeout},
		{fmt.Errorf("%w: %w", predict.ErrScoring, inference.ErrUnavailable), http.StatusServiceUnavailable},
		{fmt.Errorf("%w: %w", predict.ErrScoring, inference.ErrBadResponse), http.StatusBadGateway},
	}
	for _, tc := range cases {
		rec := do(newServer(fakeDecider{err: tc.err}), http.MethodPost, "/v1/predictions", body)
		if rec.Code != tc.want {
			t.Errorf("err %v: got %d want %d", tc.err, rec.Code, tc.want)
		}
	}
}

func TestPredictMalformedAndUnknownFields(t *testing.T) {
	h := newServer(fakeDecider{})
	if rec := do(h, http.MethodPost, "/v1/predictions", `{not json`); rec.Code != http.StatusBadRequest {
		t.Errorf("malformed: got %d", rec.Code)
	}
	if rec := do(h, http.MethodPost, "/v1/predictions", `{"agee": 30}`); rec.Code != http.StatusBadRequest {
		t.Errorf("unknown field: got %d", rec.Code)
	}
	if rec := do(h, http.MethodPost, "/v1/predictions", strings.Repeat("x", 1<<17)); rec.Code != http.StatusBadRequest {
		t.Errorf("too large: got %d", rec.Code)
	}
}

func TestMethodNotAllowedAndHealth(t *testing.T) {
	h := newServer(fakeDecider{})
	if rec := do(h, http.MethodGet, "/v1/predictions", ""); rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET predictions: got %d", rec.Code)
	}
	if rec := do(h, http.MethodGet, "/health", ""); rec.Code != http.StatusOK {
		t.Errorf("health: got %d", rec.Code)
	}
}

// Command gateway: REST API publik filter SMS (Go) di depan Python inference service.
//
//	INFERENCE_URL=http://localhost:8001 PORT=8081 go run ./cmd/gateway
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ariskaAdi/ml-bootcamp/capstone/services/gateway-go/internal/filter"
	"github.com/ariskaAdi/ml-bootcamp/capstone/services/gateway-go/internal/httpapi"
	"github.com/ariskaAdi/ml-bootcamp/capstone/services/gateway-go/internal/scorer"
)

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	timeout, err := time.ParseDuration(env("INFERENCE_TIMEOUT", "2s"))
	if err != nil {
		log.Error("invalid INFERENCE_TIMEOUT", "error", err)
		os.Exit(1)
	}
	failOpen := env("FAIL_OPEN", "true") == "true"
	client := scorer.New(env("INFERENCE_URL", "http://localhost:8001"), timeout)
	h := &httpapi.Handler{Service: &filter.Service{Scorer: client, FailOpen: failOpen}, Ready: client, Log: log}
	srv := &http.Server{Addr: ":" + env("PORT", "8081"), Handler: h.Routes(),
		ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, WriteTimeout: 10 * time.Second}

	go func() {
		log.Info("listening", "addr", srv.Addr, "inference_url", client.BaseURL, "fail_open", failOpen)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("server error", "error", err)
			os.Exit(1)
		}
	}()
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx)
}

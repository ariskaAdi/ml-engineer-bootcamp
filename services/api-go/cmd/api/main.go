// Command api adalah REST API publik (Go) yang memanggil Python inference service.
//
//	INFERENCE_URL=http://localhost:8000 PORT=8080 go run ./cmd/api
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

	"github.com/ariskaAdi/ml-bootcamp/services/api-go/internal/httpapi"
	"github.com/ariskaAdi/ml-bootcamp/services/api-go/internal/inference"
	"github.com/ariskaAdi/ml-bootcamp/services/api-go/internal/predict"
)

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	timeout, err := time.ParseDuration(env("INFERENCE_TIMEOUT", "2s"))
	if err != nil {
		logger.Error("invalid INFERENCE_TIMEOUT", "error", err)
		os.Exit(1)
	}
	client := inference.New(env("INFERENCE_URL", "http://localhost:8000"), timeout)
	h := &httpapi.Handler{Service: &predict.Service{Scorer: client}, Ready: client, Log: logger}

	srv := &http.Server{
		Addr:              ":" + env("PORT", "8080"),
		Handler:           h.Routes(),
		ReadHeaderTimeout: 5 * time.Second, // lindungi dari klien yang sangat lambat
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
	}
	go func() {
		logger.Info("listening", "addr", srv.Addr, "inference_url", client.BaseURL, "timeout", timeout.String())
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	// Graceful shutdown: selesaikan request yang sedang berjalan sebelum keluar (penting di Docker/cloud).
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx)
	logger.Info("stopped")
}

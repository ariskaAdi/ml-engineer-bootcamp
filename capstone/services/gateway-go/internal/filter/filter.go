// Package filter berisi aturan bisnis filter SMS. Tidak tahu apa-apa tentang HTTP.
package filter

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"
)

const MaxTextRunes = 1000

// QuarantineProbability: di atas ini pesan DITAHAN. Di antara threshold model dan nilai ini,
// pesan tetap dikirim tetapi diberi label peringatan — supaya pesan normal yang salah dinilai
// (false positive) tidak hilang dari pengguna.
const QuarantineProbability = 0.90

type Message struct {
	MessageID       string `json:"message_id"`
	Sender          string `json:"sender"`
	SenderIsContact bool   `json:"sender_is_contact"`
	Text            string `json:"text"`
}

type Score struct {
	Probability  float64
	IsSpam       bool // sudah memakai threshold model (precision >= 98% di validasi)
	ModelVersion string
}

type Scorer interface {
	Score(ctx context.Context, text string) (Score, error)
}

type Verdict struct {
	MessageID       string   `json:"message_id"`
	Action          string   `json:"action"` // deliver | warn | quarantine
	SpamProbability *float64 `json:"spam_probability"`
	Reason          string   `json:"reason"`
	ModelVersion    string   `json:"model_version,omitempty"`
	Degraded        bool     `json:"degraded"` // true = keputusan dibuat tanpa model
}

type ValidationError struct{ Problems []string }

func (e *ValidationError) Error() string { return "invalid input: " + strings.Join(e.Problems, "; ") }

var ErrScoring = errors.New("scoring failed")

func Validate(m Message) error {
	var p []string
	if strings.TrimSpace(m.MessageID) == "" {
		p = append(p, "message_id: wajib diisi")
	}
	if strings.TrimSpace(m.Text) == "" {
		p = append(p, "text: wajib diisi")
	}
	if n := utf8.RuneCountInString(m.Text); n > MaxTextRunes {
		p = append(p, fmt.Sprintf("text: %d karakter, maksimal %d", n, MaxTextRunes))
	}
	if !utf8.ValidString(m.Text) {
		p = append(p, "text: bukan UTF-8 yang valid")
	}
	if len(p) > 0 {
		return &ValidationError{Problems: p}
	}
	return nil
}

// Service menerapkan aturan bisnis di sekitar model.
type Service struct {
	Scorer Scorer
	// FailOpen: kalau model tidak bisa dihubungi, kirimkan pesan (deliver) dan tandai degraded,
	// daripada memblokir SEMUA SMS pengguna. Untuk filter spam, gagal-terbuka lebih aman bagi pengguna.
	FailOpen bool
}

func (s *Service) Classify(ctx context.Context, m Message) (Verdict, error) {
	if err := Validate(m); err != nil {
		return Verdict{}, err
	}
	v := Verdict{MessageID: m.MessageID}
	if m.SenderIsContact {
		v.Action, v.Reason = "deliver", "pengirim ada di daftar kontak"
		return v, nil
	}
	sc, err := s.Scorer.Score(ctx, m.Text)
	if err != nil {
		if s.FailOpen {
			v.Action, v.Reason, v.Degraded = "deliver", "model tidak tersedia (fail-open)", true
			return v, nil
		}
		return Verdict{}, fmt.Errorf("%w: %w", ErrScoring, err)
	}
	p := sc.Probability
	v.SpamProbability, v.ModelVersion = &p, sc.ModelVersion
	switch {
	case p >= QuarantineProbability:
		v.Action, v.Reason = "quarantine", "sangat mungkin spam"
	case sc.IsSpam:
		v.Action, v.Reason = "warn", "mungkin spam: dikirim dengan label peringatan"
	default:
		v.Action, v.Reason = "deliver", "bukan spam"
	}
	return v, nil
}

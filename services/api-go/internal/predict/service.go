package predict

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
)

// MaxContactsPerCampaign: aturan bisnis — jangan ganggu nasabah yang sudah terlalu sering ditelepon,
// berapa pun skor modelnya. Aturan seperti ini milik layer bisnis (Go), bukan model.
const MaxContactsPerCampaign = 10

// HighPriorityProbability: peluang di atas ini masuk antrean prioritas tinggi.
const HighPriorityProbability = 0.5

// ValidationError berisi daftar field yang salah, supaya klien bisa memperbaiki semuanya sekaligus.
type ValidationError struct{ Problems []string }

func (e *ValidationError) Error() string { return "invalid input: " + strings.Join(e.Problems, "; ") }

var (
	jobs       = []string{"admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"}
	maritals   = []string{"divorced", "married", "single", "unknown"}
	educations = []string{"basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"}
	yesNoUnk   = []string{"no", "yes", "unknown"}
	contacts   = []string{"cellular", "telephone"}
	months     = []string{"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}
	days       = []string{"mon", "tue", "wed", "thu", "fri"}
	poutcomes  = []string{"failure", "nonexistent", "success"}
)

// Validate memeriksa input SEBELUM memanggil model (lebih cepat, pesan error lebih jelas).
func Validate(r ClientRequest) error {
	var p []string
	oneOf := func(field, v string, allowed []string) {
		if !slices.Contains(allowed, v) {
			p = append(p, fmt.Sprintf("%s: %q tidak dikenal", field, v))
		}
	}
	between := func(field string, v, lo, hi float64) {
		if v < lo || v > hi {
			p = append(p, fmt.Sprintf("%s: %v di luar rentang [%v, %v]", field, v, lo, hi))
		}
	}
	if strings.TrimSpace(r.ClientID) == "" {
		p = append(p, "client_id: wajib diisi")
	}
	between("age", float64(r.Age), 17, 100)
	oneOf("job", r.Job, jobs)
	oneOf("marital", r.Marital, maritals)
	oneOf("education", r.Education, educations)
	oneOf("default", r.Default, yesNoUnk)
	oneOf("housing", r.Housing, yesNoUnk)
	oneOf("loan", r.Loan, yesNoUnk)
	oneOf("contact", r.Contact, contacts)
	oneOf("month", r.Month, months)
	oneOf("day_of_week", r.DayOfWeek, days)
	oneOf("poutcome", r.Poutcome, poutcomes)
	between("campaign", float64(r.Campaign), 1, 100)
	between("pdays", float64(r.Pdays), 0, 999)
	between("previous", float64(r.Previous), 0, 100)
	between("emp_var_rate", r.EmpVarRate, -10, 10)
	between("cons_price_idx", r.ConsPriceIdx, 80, 110)
	between("cons_conf_idx", r.ConsConfIdx, -80, 0)
	between("euribor3m", r.Euribor3m, 0, 10)
	between("nr_employed", r.NrEmployed, 4000, 6000)
	if len(p) > 0 {
		return &ValidationError{Problems: p}
	}
	return nil
}

// Service menggabungkan validasi, model, dan aturan bisnis.
type Service struct{ Scorer Scorer }

// ErrScoring dibungkus di atas error dari Scorer supaya handler bisa memetakan ke status HTTP.
var ErrScoring = errors.New("scoring failed")

func (s *Service) Decide(ctx context.Context, r ClientRequest) (Decision, error) {
	if err := Validate(r); err != nil {
		return Decision{}, err
	}
	// Aturan bisnis yang tidak butuh model: tidak perlu memanggil Python sama sekali.
	if r.Campaign >= MaxContactsPerCampaign {
		return Decision{ClientID: r.ClientID, Action: "do_not_call", Priority: "none",
			Reason: fmt.Sprintf("sudah dihubungi %d kali di kampanye ini (batas %d)", r.Campaign, MaxContactsPerCampaign)}, nil
	}
	sc, err := s.Scorer.Score(ctx, toFeatures(r))
	if err != nil {
		return Decision{}, fmt.Errorf("%w: %w", ErrScoring, err)
	}
	d := Decision{ClientID: r.ClientID, Probability: sc.Probability, ModelVersion: sc.ModelVersion}
	switch {
	case sc.Probability >= HighPriorityProbability:
		d.Action, d.Priority, d.Reason = "call", "high", "peluang tinggi"
	case sc.ModelCall:
		d.Action, d.Priority, d.Reason = "call", "medium", "di atas threshold model (20% teratas)"
	default:
		d.Action, d.Priority, d.Reason = "skip", "low", "di bawah threshold model"
	}
	return d, nil
}

func toFeatures(r ClientRequest) Features {
	return Features{Age: r.Age, Job: r.Job, Marital: r.Marital, Education: r.Education, Default: r.Default,
		Housing: r.Housing, Loan: r.Loan, Contact: r.Contact, Month: r.Month, DayOfWeek: r.DayOfWeek,
		Campaign: r.Campaign, Pdays: r.Pdays, Previous: r.Previous, Poutcome: r.Poutcome,
		EmpVarRate: r.EmpVarRate, ConsPriceIdx: r.ConsPriceIdx, ConsConfIdx: r.ConsConfIdx,
		Euribor3m: r.Euribor3m, NrEmployed: r.NrEmployed}
}

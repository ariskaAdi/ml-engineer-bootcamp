// Package predict berisi logika bisnis: validasi, aturan bisnis, dan keputusan akhir.
// Paket ini TIDAK tahu tentang HTTP; ia hanya memanggil antarmuka Scorer.
package predict

import "context"

// ClientRequest adalah input publik API Go. client_id hanya dipakai di sisi Go
// (untuk bisnis/logging) dan TIDAK dikirim ke model.
type ClientRequest struct {
	ClientID     string  `json:"client_id"`
	Age          int     `json:"age"`
	Job          string  `json:"job"`
	Marital      string  `json:"marital"`
	Education    string  `json:"education"`
	Default      string  `json:"default"`
	Housing      string  `json:"housing"`
	Loan         string  `json:"loan"`
	Contact      string  `json:"contact"`
	Month        string  `json:"month"`
	DayOfWeek    string  `json:"day_of_week"`
	Campaign     int     `json:"campaign"`
	Pdays        int     `json:"pdays"`
	Previous     int     `json:"previous"`
	Poutcome     string  `json:"poutcome"`
	EmpVarRate   float64 `json:"emp_var_rate"`
	ConsPriceIdx float64 `json:"cons_price_idx"`
	ConsConfIdx  float64 `json:"cons_conf_idx"`
	Euribor3m    float64 `json:"euribor3m"`
	NrEmployed   float64 `json:"nr_employed"`
}

// Features adalah bagian dari request yang dikirim ke model (tanpa client_id).
type Features struct {
	Age          int     `json:"age"`
	Job          string  `json:"job"`
	Marital      string  `json:"marital"`
	Education    string  `json:"education"`
	Default      string  `json:"default"`
	Housing      string  `json:"housing"`
	Loan         string  `json:"loan"`
	Contact      string  `json:"contact"`
	Month        string  `json:"month"`
	DayOfWeek    string  `json:"day_of_week"`
	Campaign     int     `json:"campaign"`
	Pdays        int     `json:"pdays"`
	Previous     int     `json:"previous"`
	Poutcome     string  `json:"poutcome"`
	EmpVarRate   float64 `json:"emp_var_rate"`
	ConsPriceIdx float64 `json:"cons_price_idx"`
	ConsConfIdx  float64 `json:"cons_conf_idx"`
	Euribor3m    float64 `json:"euribor3m"`
	NrEmployed   float64 `json:"nr_employed"`
}

// Score adalah hasil dari model.
type Score struct {
	Probability  float64
	ModelCall    bool
	Threshold    float64
	ModelVersion string
}

// Scorer = siapa pun yang bisa memberi skor (service Python, atau fake di tes).
type Scorer interface {
	Score(ctx context.Context, f Features) (Score, error)
}

// Decision adalah respons akhir API Go.
type Decision struct {
	ClientID     string  `json:"client_id"`
	Probability  float64 `json:"probability"`
	Action       string  `json:"action"`   // "call" | "skip" | "do_not_call"
	Priority     string  `json:"priority"` // "high" | "medium" | "low" | "none"
	Reason       string  `json:"reason"`
	ModelVersion string  `json:"model_version"`
}

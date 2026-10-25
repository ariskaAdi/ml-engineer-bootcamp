package predict

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func validRequest() ClientRequest {
	return ClientRequest{ClientID: "C001", Age: 35, Job: "admin.", Marital: "married", Education: "university.degree",
		Default: "no", Housing: "yes", Loan: "no", Contact: "cellular", Month: "may", DayOfWeek: "thu",
		Campaign: 1, Pdays: 999, Previous: 0, Poutcome: "nonexistent", EmpVarRate: -1.8, ConsPriceIdx: 92.893,
		ConsConfIdx: -46.2, Euribor3m: 1.313, NrEmployed: 5099.1}
}

// fakeScorer menggantikan service Python di unit test.
type fakeScorer struct {
	score  Score
	err    error
	called int
}

func (f *fakeScorer) Score(_ context.Context, _ Features) (Score, error) {
	f.called++
	return f.score, f.err
}

func TestValidateAcceptsValidRequest(t *testing.T) {
	if err := Validate(validRequest()); err != nil {
		t.Fatalf("expected valid, got %v", err)
	}
}

func TestValidateCollectsAllProblems(t *testing.T) {
	r := validRequest()
	r.ClientID = " "
	r.Age = 5
	r.Job = "astronaut"
	r.Euribor3m = 50
	err := Validate(r)
	var verr *ValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("expected ValidationError, got %v", err)
	}
	if len(verr.Problems) != 4 {
		t.Fatalf("expected 4 problems, got %d: %v", len(verr.Problems), verr.Problems)
	}
}

func TestDecidePriorities(t *testing.T) {
	cases := []struct {
		name         string
		score        Score
		wantAction   string
		wantPriority string
	}{
		{"high", Score{Probability: 0.7, ModelCall: true}, "call", "high"},
		{"medium", Score{Probability: 0.2, ModelCall: true}, "call", "medium"},
		{"low", Score{Probability: 0.05, ModelCall: false}, "skip", "low"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := &Service{Scorer: &fakeScorer{score: tc.score}}
			d, err := s.Decide(context.Background(), validRequest())
			if err != nil {
				t.Fatal(err)
			}
			if d.Action != tc.wantAction || d.Priority != tc.wantPriority {
				t.Fatalf("got %s/%s, want %s/%s", d.Action, d.Priority, tc.wantAction, tc.wantPriority)
			}
		})
	}
}

func TestContactFatigueRuleSkipsModel(t *testing.T) {
	fs := &fakeScorer{score: Score{Probability: 0.99, ModelCall: true}}
	r := validRequest()
	r.Campaign = MaxContactsPerCampaign
	d, err := (&Service{Scorer: fs}).Decide(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	if d.Action != "do_not_call" || fs.called != 0 {
		t.Fatalf("expected do_not_call without calling model, got %s (called=%d)", d.Action, fs.called)
	}
}

func TestInvalidInputDoesNotCallModel(t *testing.T) {
	fs := &fakeScorer{}
	r := validRequest()
	r.Month = "smarch"
	if _, err := (&Service{Scorer: fs}).Decide(context.Background(), r); err == nil {
		t.Fatal("expected error")
	}
	if fs.called != 0 {
		t.Fatal("model must not be called for invalid input")
	}
}

func TestScorerErrorIsWrapped(t *testing.T) {
	s := &Service{Scorer: &fakeScorer{err: errors.New("boom")}}
	_, err := s.Decide(context.Background(), validRequest())
	if !errors.Is(err, ErrScoring) || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("expected wrapped ErrScoring, got %v", err)
	}
}

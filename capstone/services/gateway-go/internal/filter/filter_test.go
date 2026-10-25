package filter

import (
	"context"
	"errors"
	"strings"
	"testing"
)

type fakeScorer struct {
	s      Score
	err    error
	called int
}

func (f *fakeScorer) Score(context.Context, string) (Score, error) { f.called++; return f.s, f.err }

func msg() Message { return Message{MessageID: "m1", Sender: "+62811", Text: "WIN a FREE prize now"} }

func TestActions(t *testing.T) {
	cases := []struct {
		name  string
		score Score
		want  string
	}{
		{"quarantine", Score{Probability: 0.97, IsSpam: true}, "quarantine"},
		{"warn", Score{Probability: 0.6, IsSpam: true}, "warn"},
		{"deliver", Score{Probability: 0.1, IsSpam: false}, "deliver"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v, err := (&Service{Scorer: &fakeScorer{s: tc.score}}).Classify(context.Background(), msg())
			if err != nil || v.Action != tc.want || v.SpamProbability == nil {
				t.Fatalf("got %+v err=%v, want %s", v, err, tc.want)
			}
		})
	}
}

func TestContactBypassesModel(t *testing.T) {
	fs := &fakeScorer{s: Score{Probability: 0.99, IsSpam: true}}
	m := msg()
	m.SenderIsContact = true
	v, _ := (&Service{Scorer: fs}).Classify(context.Background(), m)
	if v.Action != "deliver" || fs.called != 0 {
		t.Fatalf("contact should be delivered without model, got %+v called=%d", v, fs.called)
	}
}

func TestFailOpenDeliversWhenModelDown(t *testing.T) {
	v, err := (&Service{Scorer: &fakeScorer{err: errors.New("down")}, FailOpen: true}).Classify(context.Background(), msg())
	if err != nil || v.Action != "deliver" || !v.Degraded || v.SpamProbability != nil {
		t.Fatalf("expected degraded deliver, got %+v err=%v", v, err)
	}
}

func TestFailClosedReturnsError(t *testing.T) {
	_, err := (&Service{Scorer: &fakeScorer{err: errors.New("down")}, FailOpen: false}).Classify(context.Background(), msg())
	if !errors.Is(err, ErrScoring) {
		t.Fatalf("expected ErrScoring, got %v", err)
	}
}

func TestValidation(t *testing.T) {
	cases := map[string]Message{
		"empty id":   {Text: "hi"},
		"empty text": {MessageID: "m", Text: "   "},
		"too long":   {MessageID: "m", Text: strings.Repeat("é", MaxTextRunes+1)},
		"bad utf8":   {MessageID: "m", Text: string([]byte{0xff, 0xfe})},
	}
	for name, m := range cases {
		fs := &fakeScorer{}
		_, err := (&Service{Scorer: fs}).Classify(context.Background(), m)
		var verr *ValidationError
		if !errors.As(err, &verr) || fs.called != 0 {
			t.Errorf("%s: expected ValidationError without model call, got %v", name, err)
		}
	}
	// 1000 karakter multibyte masih valid (batas dihitung per karakter, bukan per byte)
	ok := Message{MessageID: "m", Text: strings.Repeat("é", MaxTextRunes)}
	if err := Validate(ok); err != nil {
		t.Errorf("1000 runes should be valid: %v", err)
	}
}

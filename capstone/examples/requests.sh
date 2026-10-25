#!/usr/bin/env bash
# Contoh request ke gateway (jalankan setelah docker compose up). Windows: jalankan dengan Git Bash.
API=${API:-http://localhost:8081}
call() { echo "--- $1"; curl -s -X POST "$API/v1/messages/classify" -H "Content-Type: application/json" -d "$2"; echo; }

call "spam jelas -> quarantine" '{"message_id":"m1","sender":"+44870","text":"URGENT! You have WON a £1000 cash prize. Call 09061701461 now to claim, T&C apply"}'
call "pesan normal -> deliver"  '{"message_id":"m2","sender":"+62811","text":"Jadi ketemu jam 7 ya? Aku tunggu di depan kantor"}'
call "dari kontak -> deliver tanpa model" '{"message_id":"m3","sender":"+62812","sender_is_contact":true,"text":"FREE entry!!! txt WIN"}'
call "zona abu-abu -> warn" '{"message_id":"m4","sender":"+62813","text":"Your daily text from me - a favour this time"}'
call "input salah -> 422" '{"message_id":"","text":""}'
call "field typo -> 400" '{"message_id":"m5","txt":"hi"}'
echo "--- ready"; curl -s "$API/ready"; echo

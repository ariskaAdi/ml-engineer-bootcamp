# Go API. Multi-stage: compile di image besar, jalankan di image kecil.
#   docker build -t bank-api ./api-go
FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY . .
# CGO_ENABLED=0 -> binary statis, bisa jalan di image tanpa libc
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/api ./cmd/api

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/api /api
ENV PORT=8080
EXPOSE 8080
USER nonroot
ENTRYPOINT ["/api"]

# Capstone Go gateway. Multi-stage: compile di image besar, jalankan di image kecil.
#   docker build -t spam-gateway ./gateway-go
FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY . .
# CGO_ENABLED=0 -> binary statis, bisa jalan di image tanpa libc
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gateway ./cmd/gateway

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/gateway /gateway
ENV PORT=8081
EXPOSE 8081
USER nonroot
ENTRYPOINT ["/gateway"]

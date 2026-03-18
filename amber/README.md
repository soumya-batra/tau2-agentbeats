# Amber instructions

## Provide env values
```bash
cp sample.env .env
```

And fill out the required values.

## Compile

```bash
docker run --rm -v "$PWD":/work -w /work ghcr.io/rdi-foundation/amber-cli:v0 compile amber-scenario.json5 --docker-compose tau2.yml
```

## Run

```bash
export $(grep -v '^#' .env | xargs) && docker compose -f tau2.yml up
```

# Deploy G Office to app.iamceo.ai (VPS + Docker)

Run it on the **same VPS** as n8n, behind the **same reverse proxy**, on the
subdomain `app.iamceo.ai`.

## 1) DNS
Add an A record: `app.iamceo.ai` → your VPS IP (same IP as `n8n.iamceo.ai`).

## 2) Get the code on the VPS
```bash
git clone <your-repo-url> goffice && cd goffice
cp .env.example .env
nano .env          # fill ANTHROPIC_API_KEY, change LOGIN_USER/PASS, set the URLs
docker compose up -d --build
```
The app now listens on `127.0.0.1:8000` on the VPS (not public yet).

## 3) Point the reverse proxy at it
Use the block that matches the proxy already serving n8n.

### A) Caddy  (add to your Caddyfile, then `caddy reload` / restart)
```
app.iamceo.ai {
    reverse_proxy 127.0.0.1:8000
}
```
Caddy gets the TLS cert automatically.

### B) Nginx + certbot
```nginx
server {
    server_name app.iamceo.ai;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;   # important for HTTPS links
        proxy_set_header Upgrade $http_upgrade;        # WebSocket (/ws)
        proxy_set_header Connection "upgrade";
    }
}
```
Then: `sudo certbot --nginx -d app.iamceo.ai`

### C) Traefik (if n8n uses Traefik)
Remove the `ports:` block from docker-compose.yml, attach the service to your
traefik network, and add these labels under the `goffice` service:
```yaml
    networks: [traefik]
    labels:
      - traefik.enable=true
      - traefik.http.routers.goffice.rule=Host(`app.iamceo.ai`)
      - traefik.http.routers.goffice.entrypoints=websecure
      - traefik.http.routers.goffice.tls.certresolver=myresolver   # use your resolver name
      - traefik.http.services.goffice.loadbalancer.server.port=8000
networks:
  traefik:
    external: true
```

### D) Nginx Proxy Manager (UI)
Add Proxy Host → Domain `app.iamceo.ai`, Forward to `127.0.0.1` port `8000`,
enable **Websockets support**, request a Let's Encrypt cert on the SSL tab.

## 4) Verify
- Open `https://app.iamceo.ai/login` → log in with your new LOGIN_USER/PASS.
- In G Office → n8n Automations, register your webhook
  `https://n8n.iamceo.ai/webhook/iamceo`; the callback auto-uses
  `https://app.iamceo.ai/api/n8n/callback` (because PUBLIC_BASE_URL is set).

## Updating later
```bash
git pull && docker compose up -d --build
```

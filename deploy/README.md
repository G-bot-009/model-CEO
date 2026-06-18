# Deploy G Office to iamceo.ai (VPS + Docker)

Run it on the **same VPS** as n8n, behind the **same reverse proxy**, on your
**root domain** `iamceo.ai` (n8n stays on the `n8n.iamceo.ai` subdomain).

## 1) DNS
Add an A record for the root: **`@` (iamceo.ai) → your VPS IP** (same IP as
`n8n.iamceo.ai`). Optional: add a `www` record too (CNAME `www` → `iamceo.ai`).

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
iamceo.ai, www.iamceo.ai {
    reverse_proxy 127.0.0.1:8000
}
```
Caddy gets the TLS cert automatically.

### B) Nginx + certbot
```nginx
server {
    server_name iamceo.ai www.iamceo.ai;
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
Then: `sudo certbot --nginx -d iamceo.ai -d www.iamceo.ai`

### C) Traefik (if n8n uses Traefik)
Remove the `ports:` block from docker-compose.yml, attach the service to your
traefik network, and add these labels under the `goffice` service:
```yaml
    networks: [traefik]
    labels:
      - traefik.enable=true
      - traefik.http.routers.goffice.rule=Host(`iamceo.ai`) || Host(`www.iamceo.ai`)
      - traefik.http.routers.goffice.entrypoints=websecure
      - traefik.http.routers.goffice.tls.certresolver=myresolver   # use your resolver name
      - traefik.http.services.goffice.loadbalancer.server.port=8000
networks:
  traefik:
    external: true
```

### D) Nginx Proxy Manager (UI)
Add Proxy Host → Domain `iamceo.ai`, Forward to `127.0.0.1` port `8000`,
enable **Websockets support**, request a Let's Encrypt cert on the SSL tab.

## 4) Verify
- Open `https://iamceo.ai/login` → log in with your new LOGIN_USER/PASS.
- In G Office → n8n Automations, register your webhook
  `https://n8n.iamceo.ai/webhook/iamceo`; the callback auto-uses
  `https://iamceo.ai/api/n8n/callback` (because PUBLIC_BASE_URL is set).

## Updating later
```bash
git pull && docker compose up -d --build
```

## Auto-deploy (push → live within ~1 min, no open ports)

Run the deploy script on a 1-minute cron. When a new commit lands on the
tracked branch, it pulls and rebuilds automatically.

```bash
cd /root/goffice
chmod +x deploy/autodeploy.sh
( crontab -l 2>/dev/null; echo "* * * * * /root/goffice/deploy/autodeploy.sh >> /var/log/goffice-deploy.log 2>&1" ) | crontab -
```
Check it works: `tail -f /var/log/goffice-deploy.log` (a deploy line appears within a minute of each push). Remove with `crontab -e` (delete the line).

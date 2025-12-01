# Testing the PWA

## Quick Start

### Development Server

```bash
# Activate environment
conda activate drinking-galaxies

# Run Streamlit
streamlit run streamlit_app.py
```

**Access**: <http://localhost:8501>

## Testing on Mobile Devices

### Option 1: Local Network (Same WiFi)

1. Find your computer's local IP:

   ```bash
   # macOS/Linux
   ifconfig | grep "inet " | grep -v 127.0.0.1
   
   # Should show something like: 192.168.1.100
   ```

2. Run Streamlit with network access:

   ```bash
   streamlit run streamlit_app.py --server.address=0.0.0.0
   ```

3. On mobile device, navigate to: `http://YOUR_IP:8501`
   - Example: `http://192.168.1.100:8501`

### Option 2: Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Deploy on <https://share.streamlit.io>
3. Access from any device
4. PWA features work automatically

### Option 3: ngrok Tunnel

```bash
# Install ngrok: https://ngrok.com/download
# Run Streamlit
streamlit run streamlit_app.py

# In another terminal
ngrok http 8501
```

Access via the https URL provided by ngrok.

## PWA Installation Testing

### iOS (Safari)

1. Open app in Safari
2. Tap Share button (box with arrow)
3. Tap "Add to Home Screen"
4. Confirm

**Expected:**

- Icon appears on home screen with 🌌 emoji
- Opens in standalone mode (no Safari UI)
- Theme color (#00D9FF) applied to status bar

### Android (Chrome)

1. Open app in Chrome
2. Tap menu (3 dots)
3. Tap "Install app" or "Add to Home Screen"
4. Confirm

**Expected:**

- Icon appears on home screen
- Opens in standalone mode
- Theme color in address bar and splash screen

## Verification Checklist

### Browser Console (Desktop)

Open Developer Tools (F12) and check:

```
✅ ServiceWorker registered: [registration object]
✅ Manifest loaded: /.streamlit/manifest.json
✅ No CORS errors
✅ Theme applied correctly
```

### PWA Features

- [ ] App is installable (browser shows install prompt/option)
- [ ] Standalone mode works (no browser chrome when launched)
- [ ] Theme color applied to system UI
- [ ] Service worker caches assets (check Network tab → offline mode)
- [ ] Icons load correctly (manifest.json icons array)

### Mobile UX

- [ ] Tap targets ≥48px (easy to tap with thumb)
- [ ] No horizontal scrolling
- [ ] Images scale properly (max-height on mobile)
- [ ] Navigation buttons accessible at bottom
- [ ] Sidebar opens smoothly
- [ ] Expanders toggle without lag
- [ ] Metrics readable in single row

### Theme

- [ ] Dark background (#0E1117) applied
- [ ] Cyan accent (#00D9FF) on primary buttons
- [ ] Text contrast is readable
- [ ] No white flashes on page load

## Troubleshooting

### Service Worker Not Registering

**Error**: `ServiceWorker registration failed`

**Fix**: Service worker requires HTTPS in production. Use:

- Streamlit Cloud (auto HTTPS)
- ngrok (provides HTTPS)
- Local development (localhost is exempt)

### Manifest Not Found

**Error**: `Failed to fetch manifest`

**Fix**: Ensure `.streamlit/manifest.json` is:

1. In the correct directory
2. Valid JSON (use <https://jsonlint.com>)
3. Served with correct MIME type (`application/json`)

### PWA Not Installable

**Requirements**:

- Must be served over HTTPS (or localhost)
- Must have valid manifest.json
- Must have registered service worker
- Must meet Chrome's "installability criteria"

**Check**: Chrome DevTools → Application → Manifest

### Theme Not Applying

**Check**:

1. `.streamlit/config.toml` exists
2. Restart Streamlit server after config changes
3. Clear browser cache
4. Verify theme section in config.toml

### Mobile Issues

**Images too large**: Check `.stImage { max-height: 60vh }` CSS
**Buttons too small**: Verify `.stButton > button { min-height: 48px }`
**Horizontal scroll**: Remove fixed-width elements, use `use_container_width=True`

## Performance Testing

### Lighthouse (Chrome DevTools)

1. Open DevTools (F12)
2. Go to "Lighthouse" tab
3. Select "Progressive Web App"
4. Click "Generate report"

**Target Scores**:

- PWA: ≥90 (installable, offline-capable)
- Performance: ≥70 (acceptable for dynamic apps)
- Accessibility: ≥90 (WCAG AA)

### Network Throttling

1. DevTools → Network tab
2. Set throttling to "Slow 3G"
3. Test app responsiveness
4. Verify spinners show during load

## Production Deployment

### Streamlit Cloud

```bash
# Already done - just push to GitHub
git add .
git commit -m "feat: mobile-first PWA UX improvements"
git push origin main
```

### Self-Hosted (with nginx)

```nginx
server {
    listen 80;
    server_name drinking-galaxies.example.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name drinking-galaxies.example.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Serve PWA files
    location /.streamlit/ {
        alias /path/to/drinking-galaxies/.streamlit/;
        types {
            application/json json;
            application/javascript js;
        }
    }
}
```

Run Streamlit:

```bash
streamlit run streamlit_app.py --server.address=127.0.0.1 --server.port=8501
```

## Next Steps

After verifying PWA works:

1. **Analytics**: Add tracking for PWA installs
2. **Monitoring**: Track service worker errors
3. **Updates**: Implement update notification when new version deployed
4. **Offline**: Enhance offline capabilities (bundle star catalog)
5. **Performance**: Optimize image loading and caching

---

**Last Updated**: 2025-12-01  
**Streamlit Version**: 1.51.0

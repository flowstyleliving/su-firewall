# Custom Domain Setup for semanticuncertainty.com

## 🚀 **Step 1: Cloudflare Pages Configuration**

### Via Cloudflare Dashboard:
1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Navigate to **Pages** → **semantic-uncertainty-dashboard**
3. Click **Custom domains** tab
4. Click **Set up a custom domain**
5. Enter: `semanticuncertainty.com`
6. Click **Continue** and follow the setup wizard

### Via Wrangler (Alternative):
```bash
# List current projects
wrangler pages project list

# Check current domains
wrangler pages deployment list --project-name semantic-uncertainty-dashboard
```

## 🌐 **Step 2: DNS Configuration**

### Required DNS Records:
```bash
# A Record (Root Domain)
Type: A
Name: @
Value: 192.0.2.1 (Cloudflare proxy)
Proxy: Enabled (Orange cloud)

# CNAME Record (WWW)
Type: CNAME
Name: www
Value: semanticuncertainty.com
Proxy: Enabled (Orange cloud)

# CNAME Record (API Subdomain)
Type: CNAME
Name: api
Value: semantic-uncertainty-runtime-physics-production.mys628.workers.dev
Proxy: Disabled (Gray cloud)
```

### Cloudflare DNS Setup:
1. Go to **DNS** → **Records**
2. Add the above records
3. Ensure SSL/TLS is set to **Full (strict)**
4. Enable **Always Use HTTPS**

## 🔒 **Step 3: SSL Certificate Setup**

### Automatic SSL (Recommended):
1. In Cloudflare dashboard, go to **SSL/TLS**
2. Set encryption mode to **Full (strict)**
3. Enable **Always Use HTTPS**
4. Enable **HSTS** (HTTP Strict Transport Security)

### Manual SSL (If needed):
```bash
# Generate SSL certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Upload to Cloudflare
# (Use Cloudflare dashboard for easier management)
```

## 🧪 **Step 4: Test Integration**

### Test API Endpoints:
```bash
# Health check
curl -s "https://semanticuncertainty.com/api/health" | jq .

# Analysis endpoint
curl -X POST "https://semanticuncertainty.com/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test prompt", "output": "Test output"}'
```

### Test Dashboard:
```bash
# Check dashboard loads
curl -s "https://semanticuncertainty.com" | head -10

# Check SSL certificate
openssl s_client -connect semanticuncertainty.com:443 -servername semanticuncertainty.com
```

## 📊 **Step 5: Performance Monitoring**

### Key Metrics to Monitor:
- **Response Time**: <50ms for API
- **Uptime**: >99.9%
- **SSL Certificate**: Valid and auto-renewing
- **DNS Propagation**: Complete within 24 hours

### Monitoring Commands:
```bash
# Check API health
curl -s "https://semanticuncertainty.com/api/health" | jq .

# Test dashboard performance
curl -w "@curl-format.txt" -o /dev/null -s "https://semanticuncertainty.com"

# Monitor SSL certificate
echo | openssl s_client -connect semanticuncertainty.com:443 -servername semanticuncertainty.com 2>/dev/null | openssl x509 -noout -dates
```

## 🔧 **Step 6: Update Configuration**

### Update API URLs:
```javascript
// In dashboard/enhanced_diagnostics_dashboard.py
self.cloudflare_config = {
    'production_url': 'https://api.semanticuncertainty.com',
    'staging_url': 'https://staging-api.semanticuncertainty.com',
    'api_key': 'your-api-key'
}
```

### Update Documentation:
```markdown
# Update all documentation files with new URLs
- API: https://api.semanticuncertainty.com
- Dashboard: https://semanticuncertainty.com
- Documentation: https://docs.semanticuncertainty.com
```

## 🚨 **Troubleshooting**

### Common Issues:
1. **DNS Not Propagated**: Wait 24-48 hours
2. **SSL Certificate Issues**: Check Cloudflare SSL settings
3. **API Not Working**: Verify CNAME records
4. **Dashboard Not Loading**: Check Pages deployment

### Debug Commands:
```bash
# Check DNS propagation
dig semanticuncertainty.com
nslookup semanticuncertainty.com

# Check SSL certificate
openssl s_client -connect semanticuncertainty.com:443

# Test API connectivity
curl -v "https://api.semanticuncertainty.com/health"
```

## ✅ **Success Criteria**

- [ ] Custom domain configured in Cloudflare Pages
- [ ] DNS records properly set up
- [ ] SSL certificate valid and auto-renewing
- [ ] API responding at new domain
- [ ] Dashboard loading correctly
- [ ] Performance metrics within targets
- [ ] Documentation updated with new URLs

## 📞 **Support**

If issues persist:
1. Check Cloudflare status page
2. Review Cloudflare logs
3. Test with different DNS servers
4. Contact Cloudflare support if needed

---

*This guide ensures a smooth transition to the custom domain with proper SSL, DNS, and performance monitoring.* 
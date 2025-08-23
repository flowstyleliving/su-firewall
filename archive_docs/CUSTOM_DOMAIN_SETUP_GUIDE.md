# Custom Domain Setup Guide

## Current Status
- ✅ Dashboard deployed: https://semantic-uncertainty-dashboard.pages.dev
- ✅ Main site deployed: https://semanticuncertainty-com.pages.dev
- ❌ Custom domain not configured: https://semanticuncertainty.com (522 error)

## Steps to Configure Custom Domain

### 1. Access Cloudflare Pages Dashboard
1. Go to https://dash.cloudflare.com
2. Navigate to "Pages" in the left sidebar
3. Click on the "semanticuncertainty-com" project

### 2. Configure Custom Domain
1. In the project dashboard, click on "Custom domains" tab
2. Click "Set up a custom domain"
3. Enter: `semanticuncertainty.com`
4. Click "Continue"

### 3. DNS Configuration
The DNS should already be configured since you mentioned you set up the 3 custom domains. Verify these DNS records exist in your Cloudflare DNS settings:

```
Type: CNAME
Name: semanticuncertainty.com
Target: semanticuncertainty-com.pages.dev
Proxy: Enabled (orange cloud)
```

```
Type: CNAME
Name: www.semanticuncertainty.com
Target: semanticuncertainty-com.pages.dev
Proxy: Enabled (orange cloud)
```

```
Type: CNAME
Name: api.semanticuncertainty.com
Target: neural-uncertainty-api.michael-account.workers.dev
Proxy: Enabled (orange cloud)
```

### 4. SSL/TLS Configuration
1. In Cloudflare dashboard, go to "SSL/TLS" section
2. Set SSL/TLS encryption mode to "Full (strict)"
3. Enable "Always Use HTTPS"

### 5. Test the Configuration
After completing the above steps, test:
```bash
curl -I https://semanticuncertainty.com
```

## Alternative: Quick Test
If you want to test immediately, you can access the working dashboard at:
https://semantic-uncertainty-dashboard.pages.dev

## Troubleshooting
- If you still get 522 errors, check that the DNS records are properly configured
- Ensure the Cloudflare Pages project is active and deployed
- Check that SSL certificates are properly provisioned (may take a few minutes)

## Next Steps
Once the custom domain is working:
1. Test the full integration
2. Update any hardcoded URLs in the codebase
3. Set up monitoring and analytics 
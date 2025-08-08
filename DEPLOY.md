# 🚀 Deploying DATect to Vercel

## Prerequisites
1. Create a Vercel account at https://vercel.com
2. Install Vercel CLI: `npm i -g vercel`

## Deployment Steps

### 1. Login to Vercel
```bash
vercel login
```

### 2. Deploy to Vercel
```bash
# From project root directory
vercel

# Follow prompts:
# - Set up and deploy? Y
# - Which scope? (select your account)
# - Link to existing project? N
# - Project name? datect-app (or your choice)
# - Directory? ./ (current directory)
# - Override settings? N
```

### 3. Deploy to Production
```bash
vercel --prod
```

## What Gets Deployed

- **Frontend**: React app built with Vite → Served from CDN
- **Backend**: FastAPI converted to serverless functions → `/api/*` routes
- **Static Assets**: Automatically optimized and cached

## Environment Variables (Optional)

If you need to add API keys or secrets:

1. Go to Vercel Dashboard → Your Project → Settings → Environment Variables
2. Add variables like:
   - `SATELLITE_API_KEY`
   - `DATABASE_URL`
   - etc.

## File Structure Created

```
api/
  index.py           # Serverless API handler
  requirements.txt   # Python dependencies
frontend/
  .env.production   # Production API endpoint
vercel.json         # Deployment configuration
.gitignore          # Updated with Vercel files
```

## Access Your App

After deployment:
- **Production**: https://your-app.vercel.app
- **API Endpoints**: https://your-app.vercel.app/api/*
- **Dashboard**: https://vercel.com/dashboard

## Automatic Deployments

Push to GitHub for automatic deployments:
```bash
git add .
git commit -m "Deploy to Vercel"
git push
```

Vercel will automatically deploy on every push to main branch.

## Troubleshooting

- **Build errors**: Check `vercel.json` configuration
- **API timeout**: Increase `maxDuration` in vercel.json (max 60s on free tier)
- **Large dependencies**: Some ML libraries may exceed size limits - consider using lighter alternatives

## Notes

- Free tier includes 100GB bandwidth/month
- Serverless functions have 10s timeout (free) or 60s (pro)
- Python runtime supports up to 250MB deployment size
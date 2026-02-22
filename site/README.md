# 2048 AI Leaderboard Site

Static website with Chart.js visualizations for 2048 AI benchmark results.

## Local Development

Open `index.html` directly in a browser, or serve locally:

```bash
# Using Python
python -m http.server 8000 -d site

# Using Node.js
npx serve site
```

## Deployment

### Vercel

The site is configured for Vercel deployment. The root `vercel.json` handles this.

### GitHub Pages

1. Go to Settings > Pages
2. Set Source to "Deploy from a branch"
3. Select `main` branch and `/site` folder
4. Save

## Features

- Bar chart ranking agents by average score
- Line chart showing score range (min/avg/max)
- Doughnut chart for max tile distribution
- Radar chart for multi-dimensional agent comparison
- Responsive design
- Dark mode support (follows system preference)
- Sortable leaderboard table

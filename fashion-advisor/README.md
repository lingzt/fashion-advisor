# Fashion Advisor - AI Fashion Recommendation System

AI-powered fashion recommendations using RAG (Retrieval Augmented Generation).

## Features

- 🔍 Natural language fashion search
- 🖼️ Visual recommendations from fashion database
- 🎨 Color and style filtering
- 🔐 Password protection

## Deployment

**URL:** lingtoby.com/fashion-advisor

### Deploy to Vercel

```bash
cd fashion-advisor
vercel --prod
```

### Add Custom Domain

1. Vercel Dashboard → Settings → Domains
2. Add: `lingtoby.com`
3. Configure path-based routing to `/fashion-advisor`

## Local Development

```bash
cd fashion-advisor
npm install
npm run dev
```

Open http://localhost:3000

## API

Backend API: `https://fashion-rag-api.onrender.com`

## Tech Stack

- Next.js 14
- React
- OpenAI GPT-3.5
- ChromaDB Vector Database

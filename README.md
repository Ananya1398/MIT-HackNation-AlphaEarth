# AlphaEarth Risk and Insurance Payout Compass

## Project Description

A real-time multi-hazard risk assessment platform that generates Low/Medium/High risk scores for flood, hurricane, and wildfire exposure across U.S. counties. The system uses live climate data from Open-Meteo ERA5, historical FEMA NFIP claims, and a linear regression ML model to predict insurance payouts based on climate markers.

**Key Features:**
- Multi-hazard risk scoring (flood, hurricane, wildfire)
- County-level risk visualization on Google Maps
- ML-based insurance payout predictions
- Interactive dashboard with CSV export

## Setup Instructions

### Backend
```bash
git clone https://github.com/shivani1805/AlphaEarth-HackNation-MIT.git
cd AlphaEarth-HackNation-MIT
npm install
pip install pandas numpy requests tqdm
npm start
Server runs on http://localhost:5001
```

### Frontend
```bash
cd frontend/frontend/local-hazard-map
npm install
npm run dev
Frontend runs on http://localhost:5173
```

## Dependencies
### Backend (Node.js)
express ^5.1.0

cors ^2.8.5

csv-parser ^3.2.0

dotenv ^17.2.3

### Python
pandas

numpy

requests

tqdm

### Frontend
React 18+

TypeScript

Vite

@react-google-maps/api

Tailwind CSS

shadcn/ui components

### Team Members
Shivani Sharma
Ananya Asthana
Orijeet Mukherjee
Jane Millward

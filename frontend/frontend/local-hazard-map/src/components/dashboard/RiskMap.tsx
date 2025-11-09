import { Card } from "@/components/ui/card";
import { Globe, Key } from "lucide-react";
import { useState, useEffect } from "react";
import worldMapImage from "@/assets/world-map.png";
import GoogleMapView from "./GoogleMapView";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";


interface RiskMapProps {
  coordinates: { lat: number; lng: number } | null;
  isAnalyzing: boolean;
  disasterType: "flood" | "hurricane";
  startDate?: Date;
  endDate?: Date;
  onRiskDataUpdate?: (data: Record<string, number>) => void;
}

interface RegionData {
  id: string;
  name: string;
  riskScore: number;
  expectedClaim: number;
  polygon: [number, number][];
}

interface CityData {
  name: string;
  country: string;
  lat: number;
  lng: number;
  riskScore: number;
  expectedClaim: number;
}

const RiskMap = ({ coordinates, isAnalyzing, disasterType, startDate, endDate, onRiskDataUpdate }: RiskMapProps) => {
  const [hoveredRegion, setHoveredRegion] = useState<RegionData | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [userLocation, setUserLocation] = useState<{ lat: number; lng: number } | null>(null);
  const [showUserTooltip, setShowUserTooltip] = useState(false);
  const [regions, setRegions] = useState<RegionData[]>([]);
  const [mapDisasterType, setMapDisasterType] = useState<"flood" | "hurricane">("flood");

  const apiKey = "AIzaSyARKE4j2byp70oP0qM8DS05kNFLRZeTizU";

  // Get user's location on mount
  useEffect(() => {
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          });
        },
        (error) => {
          console.log("Geolocation error:", error.message);
        }
      );
    }
  }, []);

  // Generate random regions when coordinates are set
  useEffect(() => {
    if (coordinates && isAnalyzing) {
      generateRegions(coordinates);
    } else {
      setRegions([]);
    }
  }, [coordinates, isAnalyzing]);

  const generateRegions = (center: { lat: number; lng: number }) => {
    const newRegions: RegionData[] = [];
    const numRegions = 8; // Number of regions around the center
    const radiusInDegrees = 0.9; // Approximately 100km (1 degree ≈ 111km)

    for (let i = 0; i < numRegions; i++) {
      const angle = (i * 360) / numRegions;
      const riskScore = 0; // Default risk score
      const expectedClaim = 0; // Default expected claim
      
      // Create polygon points for each region (pie slice)
      const polygon: [number, number][] = [
        [center.lat, center.lng], // Center point
      ];
      
      const startAngle = angle - 22.5;
      const endAngle = angle + 22.5;
      
      // Create arc points
      for (let a = startAngle; a <= endAngle; a += 5) {
        const rad = (a * Math.PI) / 180;
        const lat = center.lat + radiusInDegrees * Math.cos(rad);
        const lng = center.lng + radiusInDegrees * Math.sin(rad);
        polygon.push([lat, lng]);
      }
      
      newRegions.push({
        id: `region-${i}`,
        name: `Region ${String.fromCharCode(65 + i)}`,
        riskScore,
        expectedClaim,
        polygon,
      });
    }
    
    setRegions(newRegions);
  };

  // Major world cities with initial risk score of 0
  const cities: CityData[] = [
    { name: "New York", country: "USA", lat: 40.7128, lng: -74.0060, riskScore: 0, expectedClaim: 0 },
    { name: "Los Angeles", country: "USA", lat: 34.0522, lng: -118.2437, riskScore: 0, expectedClaim: 0 },
    { name: "London", country: "UK", lat: 51.5074, lng: -0.1278, riskScore: 0, expectedClaim: 0 },
    { name: "Paris", country: "France", lat: 48.8566, lng: 2.3522, riskScore: 0, expectedClaim: 0 },
    { name: "Tokyo", country: "Japan", lat: 35.6762, lng: 139.6503, riskScore: 0, expectedClaim: 0 },
    { name: "Beijing", country: "China", lat: 39.9042, lng: 116.4074, riskScore: 0, expectedClaim: 0 },
    { name: "Mumbai", country: "India", lat: 19.0760, lng: 72.8777, riskScore: 0, expectedClaim: 0 },
    { name: "Dubai", country: "UAE", lat: 25.2048, lng: 55.2708, riskScore: 0, expectedClaim: 0 },
    { name: "Sydney", country: "Australia", lat: -33.8688, lng: 151.2093, riskScore: 0, expectedClaim: 0 },
    { name: "São Paulo", country: "Brazil", lat: -23.5505, lng: -46.6333, riskScore: 0, expectedClaim: 0 },
    { name: "Mexico City", country: "Mexico", lat: 19.4326, lng: -99.1332, riskScore: 0, expectedClaim: 0 },
    { name: "Cairo", country: "Egypt", lat: 30.0444, lng: 31.2357, riskScore: 0, expectedClaim: 0 },
    { name: "Moscow", country: "Russia", lat: 55.7558, lng: 37.6173, riskScore: 0, expectedClaim: 0 },
    { name: "Singapore", country: "Singapore", lat: 1.3521, lng: 103.8198, riskScore: 0, expectedClaim: 0 },
    { name: "Toronto", country: "Canada", lat: 43.6532, lng: -79.3832, riskScore: 0, expectedClaim: 0 },
    { name: "Berlin", country: "Germany", lat: 52.5200, lng: 13.4050, riskScore: 0, expectedClaim: 0 },
    { name: "Hong Kong", country: "China", lat: 22.3193, lng: 114.1694, riskScore: 0, expectedClaim: 0 },
    { name: "Istanbul", country: "Turkey", lat: 41.0082, lng: 28.9784, riskScore: 0, expectedClaim: 0 },
    { name: "Seoul", country: "South Korea", lat: 37.5665, lng: 126.9780, riskScore: 0, expectedClaim: 0 },
    { name: "Bangkok", country: "Thailand", lat: 13.7563, lng: 100.5018, riskScore: 0, expectedClaim: 0 },
    { name: "Buenos Aires", country: "Argentina", lat: -34.6037, lng: -58.3816, riskScore: 0, expectedClaim: 0 },
    { name: "Lagos", country: "Nigeria", lat: 6.5244, lng: 3.3792, riskScore: 0, expectedClaim: 0 },
    { name: "Jakarta", country: "Indonesia", lat: -6.2088, lng: 106.8456, riskScore: 0, expectedClaim: 0 },
    { name: "Johannesburg", country: "South Africa", lat: -26.2041, lng: 28.0473, riskScore: 0, expectedClaim: 0 },
    { name: "Madrid", country: "Spain", lat: 40.4168, lng: -3.7038, riskScore: 0, expectedClaim: 0 },
  ];

  // Convert lat/lng to SVG coordinates (Equirectangular projection)
  const latLngToXY = (lat: number, lng: number) => {
    // Equirectangular projection formula
    // X maps from -180 to 180 longitude
    // Y maps from 90 to -90 latitude (inverted because SVG Y increases downward)
    const x = ((lng + 180) / 360) * 100;
    const y = ((90 - lat) / 180) * 100;
    return { x, y };
  };

  const getRiskColor = (riskScore: number) => {
    if (riskScore >= 8) return "hsl(var(--risk-high))";
    if (riskScore >= 6) return "hsl(var(--risk-medium-high))";
    if (riskScore >= 4) return "hsl(var(--risk-medium))";
    if (riskScore >= 2) return "hsl(var(--risk-medium-low))";
    return "hsl(var(--risk-low))";
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    setMousePos({ x: e.clientX, y: e.clientY });
  };

  return (
    <Card className="p-6 bg-card border-border relative overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Globe className="h-5 w-5 text-primary" />
          <h3 className="text-lg font-semibold">
            {isAnalyzing && coordinates ? "Location Risk Map" : "Global Risk Map"}
          </h3>
        </div>
      
          <ToggleGroup 
            type="single" 
            value={mapDisasterType}
            onValueChange={(value) => value && setMapDisasterType(value as "flood" | "hurricane")}
          >
            <ToggleGroupItem value="flood" aria-label="Flood">
              Flood
            </ToggleGroupItem>
            <ToggleGroupItem value="hurricane" aria-label="Hurricane">
              Hurricane
            </ToggleGroupItem>
          </ToggleGroup>
      
      </div>


      <div 
        className="relative h-[500px] bg-secondary/30 rounded-lg overflow-hidden"
        onMouseMove={handleMouseMove}
      >
        {/* Show Google Maps when API key is provided */}
        {apiKey ? (
          <GoogleMapView
            apiKey={apiKey}
            coordinates={coordinates}
            userLocation={userLocation}
            isAnalyzing={isAnalyzing}
            disasterType={mapDisasterType}
            mapDisasterType={mapDisasterType}
            startDate={startDate}
            endDate={endDate}
            regions={regions}
            onHoverRegion={setHoveredRegion}
            getRiskColor={getRiskColor}
            onRiskDataUpdate={onRiskDataUpdate}
          />
        ) : (
          <>
            {/* World map background */}
            <img 
              src={worldMapImage} 
              alt="World Map" 
              className="absolute inset-0 w-full h-full object-fill opacity-40 mix-blend-luminosity"
            />
            
            <svg 
              className="absolute inset-0 w-full h-full transition-all duration-1000 ease-in-out" 
              viewBox={
                coordinates && isAnalyzing
                  ? (() => {
                      const { x, y } = latLngToXY(coordinates.lat, coordinates.lng);
                      const zoomWidth = 15;
                      const zoomHeight = 15;
                      return `${x - zoomWidth/2} ${y - zoomHeight/2} ${zoomWidth} ${zoomHeight}`;
                    })()
                  : "0 0 100 100"
              }
              preserveAspectRatio="xMidYMid meet"
            >
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map((pos) => (
            <g key={`grid-${pos}`}>
              <line x1={pos} y1="0" x2={pos} y2="100" stroke="hsl(var(--border))" strokeWidth="0.15" opacity="0.2" />
              <line x1="0" y1={pos} x2="100" y2={pos} stroke="hsl(var(--border))" strokeWidth="0.15" opacity="0.2" />
            </g>
          ))}

          {/* 100km radius circle and regions */}
          {coordinates && regions.length > 0 && (() => {
            const center = latLngToXY(coordinates.lat, coordinates.lng);
            const radiusInDegrees = 0.9;
            const radiusX = (radiusInDegrees / 360) * 100;
            const radiusY = (radiusInDegrees / 180) * 100;
            
            return (
              <g>
                {/* Region polygons */}
                {regions.map((region) => {
                  const points = region.polygon.map(([lat, lng]) => {
                    const { x, y } = latLngToXY(lat, lng);
                    return `${x},${y}`;
                  }).join(' ');
                  
                  return (
                    <polygon
                      key={region.id}
                      points={points}
                      fill={getRiskColor(region.riskScore)}
                      opacity="0.6"
                      stroke="hsl(var(--border))"
                      strokeWidth="0.15"
                      className="transition-all duration-300 cursor-pointer hover:opacity-80"
                      onMouseEnter={() => setHoveredRegion(region)}
                      onMouseLeave={() => setHoveredRegion(null)}
                    />
                  );
                })}

                {/* 100km radius circle outline */}
                <ellipse
                  cx={center.x}
                  cy={center.y}
                  rx={radiusX}
                  ry={radiusY}
                  fill="none"
                  stroke="hsl(var(--primary))"
                  strokeWidth="0.3"
                  strokeDasharray="1 0.5"
                  opacity="0.6"
                />
                
                {/* Center marker for searched location */}
                <circle
                  cx={center.x}
                  cy={center.y}
                  r="0.8"
                  fill="hsl(var(--accent))"
                  stroke="hsl(var(--background))"
                  strokeWidth="0.2"
                />
              </g>
            );
          })()}

          {/* User location marker */}
          {userLocation && (() => {
            const { x, y } = latLngToXY(userLocation.lat, userLocation.lng);
            return (
              <g
                className="cursor-pointer"
                onMouseEnter={() => setShowUserTooltip(true)}
                onMouseLeave={() => setShowUserTooltip(false)}
              >
                {/* Pulsing outer ring */}
                <circle
                  cx={x}
                  cy={y}
                  r="2"
                  fill="none"
                  stroke="hsl(var(--primary))"
                  strokeWidth="0.2"
                  opacity="0.6"
                  className="animate-pulse"
                />
                {/* Inner glow */}
                <circle
                  cx={x}
                  cy={y}
                  r="1.2"
                  fill="hsl(var(--primary))"
                  opacity="0.3"
                />
                {/* Main marker */}
                <circle
                  cx={x}
                  cy={y}
                  r="0.8"
                  fill="hsl(var(--primary))"
                  stroke="hsl(var(--background))"
                  strokeWidth="0.2"
                />
                {/* Label */}
                <text
                  x={x}
                  y={y - 2.5}
                  textAnchor="middle"
                  className="text-[0.8px] font-semibold fill-primary"
                  style={{ fontSize: '2.5px' }}
                >
                  You're here
                </text>
              </g>
            );
          })()}
        </svg>
          </>
        )}

        {/* User location tooltip */}
        {showUserTooltip && userLocation && (
          <div
            className="fixed bg-primary/90 backdrop-blur-sm border border-primary rounded-lg p-3 shadow-lg z-50 pointer-events-none"
            style={{
              left: `${mousePos.x + 20}px`,
              top: `${mousePos.y + 20}px`,
            }}
          >
            <h4 className="font-semibold text-primary-foreground mb-1">Your Location</h4>
            <div className="space-y-1 text-sm text-primary-foreground/90">
              <div className="flex justify-between gap-4">
                <span>Coordinates:</span>
                <span className="font-mono text-xs">{userLocation.lat.toFixed(4)}°, {userLocation.lng.toFixed(4)}°</span>
              </div>
            </div>
          </div>
        )}

        {/* Region tooltip */}
        {hoveredRegion && (
          <div
            className="fixed bg-card border border-border rounded-lg p-4 shadow-lg z-50 pointer-events-none"
            style={{
              left: `${mousePos.x + 20}px`,
              top: `${mousePos.y + 20}px`,
            }}
          >
            <h4 className="font-semibold mb-2">{hoveredRegion.name}</h4>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between gap-6">
                <span className="text-muted-foreground">Risk Score:</span>
                <span className="font-medium" style={{ color: getRiskColor(hoveredRegion.riskScore) }}>
                  {hoveredRegion.riskScore.toFixed(1)}
                </span>
              </div>
              <div className="flex justify-between gap-6">
                <span className="text-muted-foreground">Expected Claim:</span>
                <span className="font-medium text-accent">${hoveredRegion.expectedClaim.toLocaleString()}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default RiskMap;

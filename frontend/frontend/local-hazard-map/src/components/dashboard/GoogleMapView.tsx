import { GoogleMap, useJsApiLoader, OverlayView } from "@react-google-maps/api";
import { useEffect, useState, useRef } from "react";

interface GoogleMapViewProps {
  apiKey: string;
  coordinates: { lat: number; lng: number } | null;
  userLocation: { lat: number; lng: number } | null;
  isAnalyzing: boolean;
  disasterType: "flood" | "hurricane";
  mapDisasterType?: "flood" | "hurricane";
  startDate?: Date;
  endDate?: Date;
  regions: Array<{
    id: string;
    name: string;
    riskScore: number;
    expectedClaim: number;
    polygon: [number, number][];
  }>;
  onHoverRegion: (region: any) => void;
  getRiskColor: (score: number) => string;
  onRiskDataUpdate?: (data: Record<string, number>) => void;
}
interface CountyRiskData {
  county: string;
  flood_category: "Low" | "Medium" | "High";
  hurricane_category: "Low" | "Medium" | "High";
  wildfire_category: "Low" | "Medium" | "High";
}
const GoogleMapView = ({
  apiKey,
  coordinates,
  userLocation,
  isAnalyzing,
  disasterType,
  mapDisasterType = "flood" ,
  startDate,
  endDate,
  regions,
  onHoverRegion,
  getRiskColor,
  onRiskDataUpdate,
}: GoogleMapViewProps) => {
  const mapRef = useRef<google.maps.Map | null>(null);
  const [countyLabels, setCountyLabels] = useState<Array<{ 
    id: string; 
    name: string;
    center: { lat: number; lng: number }; 
    riskScore: number 
  }>>([]);
  const [hoveredCounty, setHoveredCounty] = useState<{ name: string; lat: number; lng: number } | null>(null);
  const [countyRiskData, setCountyRiskData] = useState<Map<string, CountyRiskData>>(new Map());


  const { isLoaded } = useJsApiLoader({
    id: 'google-map-script',
    googleMapsApiKey: apiKey,
  });

  // Resolve CSS variable hsl(var(--token)) to concrete hsl() for Google Maps styles
  const resolveColor = (color: string) => {
    const match = color.match(/^hsl\(var\(--([^)]+)\)\)$/);
    if (match) {
      const varName = `--${match[1]}`;
      const raw = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
      if (raw) return `hsl(${raw})`;
    }
    return color;
  };

  // Determine map center - use coordinates if analyzing, otherwise user location or default
  const mapCenter = isAnalyzing && coordinates 
    ? coordinates 
    : userLocation 
    ? userLocation 
    : { lat: 42.3601, lng: -71.0589 }; // Default to Boston

  const onMapLoad = (map: google.maps.Map) => {
    mapRef.current = map;
    
    // Load and style county boundaries when analyzing
    if (isAnalyzing && coordinates) {
      loadCountyBoundaries(map, coordinates);
    }
  };

  const loadCountyBoundaries = async (map: google.maps.Map, center: { lat: number; lng: number }) => {
    try {
      // Load US counties GeoJSON from public source
      const response = await fetch('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json');
      const geojson = await response.json();
      
      // First, determine which state the center point is in
      let targetState: string | null = null;
      
      for (const feature of geojson.features) {
        const coords = feature.geometry.coordinates;
        
        // Check if the center point is within this county's polygon
        const isInside = (point: [number, number], polygon: number[][][]) => {
          for (const ring of polygon) {
            let inside = false;
            for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
              const xi = ring[i][0], yi = ring[i][1];
              const xj = ring[j][0], yj = ring[j][1];
              
              const intersect = ((yi > point[1]) !== (yj > point[1])) &&
                (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi);
              if (intersect) inside = !inside;
            }
            if (inside) return true;
          }
          return false;
        };
        
        if (feature.geometry.type === 'Polygon') {
          if (isInside([center.lng, center.lat], [coords])) {
            targetState = feature.properties.STATE;
            break;
          }
        } else if (feature.geometry.type === 'MultiPolygon') {
          for (const polygon of coords) {
            if (isInside([center.lng, center.lat], [polygon])) {
              targetState = feature.properties.STATE;
              break;
            }
          }
          if (targetState) break;
        }
      }
      
      // If no state found, fall back to the first county's state in the area
      if (!targetState) {
        const radiusInDegrees = 0.1;
        const minLat = center.lat - radiusInDegrees;
        const maxLat = center.lat + radiusInDegrees;
        const minLng = center.lng - radiusInDegrees;
        const maxLng = center.lng + radiusInDegrees;
        
        for (const feature of geojson.features) {
          const coords = feature.geometry.coordinates;
          if (feature.geometry.type === 'Polygon') {
            for (const coord of coords[0]) {
              const [lng, lat] = coord;
              if (lat >= minLat && lat <= maxLat && lng >= minLng && lng <= maxLng) {
                targetState = feature.properties.STATE;
                break;
              }
            }
          }
          if (targetState) break;
        }
      }

      // Filter and add features to map - only from the target state
      const countyRiskScores = new Map<string, number>();
      const riskByCountyName: Record<string, number> = {};
      const labels: Array<{ id: string; name: string; center: { lat: number; lng: number }; riskScore: number }> = [];
      
      geojson.features.forEach((feature: any) => {
        // Only process counties from the target state
        if (feature.properties.STATE !== targetState) return;
        
        const coords = feature.geometry.coordinates;
        let totalLat = 0;
        let totalLng = 0;
        let pointCount = 0;
        
        // Calculate center
        if (feature.geometry.type === 'Polygon') {
          coords[0].forEach((coord: number[]) => {
            const [lng, lat] = coord;
            totalLat += lat;
            totalLng += lng;
            pointCount++;
          });
        } else if (feature.geometry.type === 'MultiPolygon') {
          coords.forEach((polygon: any) => {
            polygon[0].forEach((coord: number[]) => {
              const [lng, lat] = coord;
              totalLat += lat;
              totalLng += lng;
              pointCount++;
            });
          });
        }
        
        if (pointCount > 0) {
          // Generate risk score based on disaster type
          let riskScore: number;
          const randomBase = Math.random();
          
          // Different risk distributions for different disasters
          if (disasterType === "flood") {
            // Flood: moderate to high risk
            riskScore = 4 + randomBase * 6; // 4-10 range
          } else if (disasterType === "hurricane") {
            // Hurricane: generally high risk
            riskScore = 6 + randomBase * 4; // 6-10 range
          } else {
            // Wildfire: varied risk
            riskScore = 2 + randomBase * 8; // 2-10 range
          }
          
          const fips = feature.id || feature.properties?.GEO_ID || `county-${Math.random()}`;
          const countyName = feature.properties?.NAME;
          
          countyRiskScores.set(fips, riskScore);
          
          // Store by county name for CSV export
          if (countyName) {
            riskByCountyName[countyName] = riskScore;
          }
          
          // Calculate centroid
          const centerLat = totalLat / pointCount;
          const centerLng = totalLng / pointCount;
          
          labels.push({
            id: fips,
            name: countyName || 'Unknown',
            center: { lat: centerLat, lng: centerLng },
            riskScore
          });
          
          // Add feature to map with risk score property
          const addedFeature = map.data.addGeoJson(feature);
          addedFeature.forEach(f => {
            f.setProperty('riskScore', riskScore);
            f.setProperty('fips', fips);
            f.setProperty('NAME', countyName);
          });
        }
      });

      setCountyLabels(labels);

      // Pass risk data to parent for CSV download
      if (onRiskDataUpdate && Object.keys(riskByCountyName).length > 0) {
        onRiskDataUpdate(riskByCountyName);
      }

      // Send county data to API
      const countyDataForApi = labels.map(label => ({
        county_name: label.name,
        centroid_coordinates: {
          latitude: label.center.lat,
          longitude: label.center.lng
        }
      }));

      // Send to API endpoint
      try {
        const response = await fetch('http://localhost:5001/api/county-risk', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            counties: countyDataForApi,
            disaster_type: disasterType,
            start_date: startDate?.toISOString(),
            end_date: endDate?.toISOString(),
            timestamp: new Date().toISOString()
          })
        });
       if (response.ok) {
          const riskData: CountyRiskData[] = await response.json();
          console.log('Received risk data from API:', riskData);
          
          // Store risk data by county name
         const riskMap = new Map<string, CountyRiskData>();
            riskData.forEach(item => {
              riskMap.set(item.county.trim().toLowerCase(), item);
            });
            setCountyRiskData(riskMap);

        }
      } catch (error) {
        console.error('Error fetching county risk data from API:', error);

      }

      // Style the features based on risk scores
      // map.data.setStyle((feature) => {
      //   const riskScore = feature.getProperty('riskScore') as number;
      //   return {
      //     fillColor: resolveColor(getRiskColor(riskScore)),
      //     fillOpacity: 0.65,
      //     strokeColor: '#ffffff',
      //     strokeOpacity: 0.9,
      //     strokeWeight: 2,
      //   };
      // });
      styleMapFeatures(map);


      // Add hover effects
      map.data.addListener('mouseover', (event: google.maps.Data.MouseEvent) => {
        map.data.overrideStyle(event.feature, {
          strokeWeight: 3,
          fillOpacity: 0.85,
        });
        
        const countyName = event.feature.getProperty('NAME') as string;
        const latLng = event.latLng;
        if (countyName && latLng) {
          setHoveredCounty({ 
            name: countyName, 
            lat: latLng.lat(), 
            lng: latLng.lng() 
          });
        }
      });

      map.data.addListener('mouseout', () => {
        map.data.revertStyle();
        setHoveredCounty(null);
      });

    } catch (error) {
      console.error('Error loading county boundaries:', error);
    }
  };
const getCategoryColor = (category: "Low" | "Medium" | "High") => {
    switch (category) {
      case "High":
        return "hsl(var(--risk-high))"; 
      case "Medium":
        return "hsl(var(--risk-medium))"; 
      case "Low":
        return "hsl(var(--risk-low))";  
      default:
        return "hsl(var(--muted))"; 
    }
  };

  // const styleMapFeatures = (map: google.maps.Map) => {
  //   map.data.setStyle((feature) => {
  //     const countyName = feature.getProperty('name') as string;
  //     const riskData = countyRiskData.get(countyName);
      
  //     let fillColor: string;
  //     if (riskData) {

  //       const category = mapDisasterType === "flood" 
  //         ? riskData.flood_category 
  //         : riskData.hurricane_category;
  //       fillColor = resolveColor(getCategoryColor(category));
  //     } else {

  //       const riskScore = feature.getProperty('riskScore') as number;
  //       fillColor = resolveColor(getRiskColor(riskScore));
  //     }
      
  //     return {
  //       fillColor,
  //       fillOpacity: 0.65,
  //       strokeColor: '#ffffff',
  //       strokeOpacity: 0.9,
  //       strokeWeight: 2,
  //     };
  //   });
  // };
  const styleMapFeatures = (map: google.maps.Map) => {
  map.data.setStyle((feature) => {
    const countyName = (feature.getProperty("NAME") as string || "").trim().toLowerCase();
    const riskData = countyRiskData.get(countyName);

    let fillColor: string;

    if (riskData) {
      const category = mapDisasterType === "flood"
        ? riskData.flood_category
        : riskData.hurricane_category;

      fillColor = resolveColor(getCategoryColor(category));
    } else {
      const riskScore = feature.getProperty("riskScore") as number;
      fillColor = resolveColor(getRiskColor(riskScore));
    }

    return {
      fillColor,
      fillOpacity: 0.65,
      strokeColor: "#ffffff",
      strokeOpacity: 0.9,
      strokeWeight: 2,
    };
  });
};


  useEffect(() => {
    if (mapRef.current && !isAnalyzing) {
      styleMapFeatures(mapRef.current);
    }
  }, [mapDisasterType, countyRiskData]);
  // Update county boundaries when analyzing changes
  useEffect(() => {
    if (mapRef.current && isAnalyzing && coordinates) {
      // Clear existing data
      mapRef.current.data.forEach((feature) => {
        mapRef.current?.data.remove(feature);
      });
      setCountyLabels([]);
      loadCountyBoundaries(mapRef.current, coordinates);
    } else if (mapRef.current && !isAnalyzing) {
      // Clear county boundaries when not analyzing
      mapRef.current.data.forEach((feature) => {
        mapRef.current?.data.remove(feature);
      });
      setCountyLabels([]);
    }
  }, [isAnalyzing, coordinates]);

  if (!isLoaded) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-secondary/30 rounded-lg">
        <p className="text-muted-foreground">Loading map...</p>
      </div>
    );
  }

  return (
    <GoogleMap
      mapContainerStyle={{ width: '100%', height: '100%' }}
      center={mapCenter}
      zoom={isAnalyzing ? 9 : 12}
      options={{
        disableDefaultUI: false,
        zoomControl: true,
        streetViewControl: false,
        mapTypeControl: true,
      }}
      onLoad={onMapLoad}
    >
    
      {hoveredCounty && (
        <OverlayView
          position={{ lat: hoveredCounty.lat, lng: hoveredCounty.lng }}
          mapPaneName={OverlayView.OVERLAY_MOUSE_TARGET}
        >
          <div
            style={{
              background: "rgba(0, 0, 0, 0.9)",
              color: "white",
              padding: "12px 18px",
              borderRadius: "8px",
              fontWeight: "600",
              fontSize: "14px",
              transform: "translate(-50%, -120%)",
              whiteSpace: "nowrap",
              pointerEvents: "none",
              boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
              border: "1px solid rgba(255, 255, 255, 0.2)",
              display: "inline-block",
              minWidth: "fit-content",
            }}
          >
            {hoveredCounty.name} County
          </div>
        </OverlayView>
      )}

      {/* User location marker - always show when available */}
      {userLocation && (
        <OverlayView
          position={{ lat: userLocation.lat, lng: userLocation.lng }}
          mapPaneName={OverlayView.OVERLAY_MOUSE_TARGET}
        >
          <div
            style={{
              transform: "translate(-50%, -100%)",
            }}
          >
            <svg
              width="24"
              height="30"
              viewBox="0 0 32 40"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              style={{
                filter: "drop-shadow(0 2px 8px rgba(0,0,0,0.4))"
              }}
            >
              <path
                d="M16 0C9.37258 0 4 5.37258 4 12C4 20.5 16 40 16 40C16 40 28 20.5 28 12C28 5.37258 22.6274 0 16 0Z"
                fill="hsl(var(--primary))"
                stroke="white"
                strokeWidth="2"
              />
              <circle
                cx="16"
                cy="12"
                r="5"
                fill="white"
              />
            </svg>
          </div>
        </OverlayView>
      )}
    </GoogleMap>
  );
};

export default GoogleMapView;

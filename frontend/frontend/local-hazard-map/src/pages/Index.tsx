import { useState } from "react";
import { MapPin } from "lucide-react";
import DashboardHeader from "@/components/dashboard/DashboardHeader";
import InputForm from "@/components/dashboard/InputForm";
import RiskMap from "@/components/dashboard/RiskMap";
import RiskLegend from "@/components/dashboard/RiskLegend";
import DownloadCsvButton from "@/components/dashboard/DownloadCsvButton";

const Index = () => {
  const [startDate, setStartDate] = useState<Date>();
  const [endDate, setEndDate] = useState<Date>();
  const [location, setLocation] = useState("");
  const [coordinates, setCoordinates] = useState<{ lat: number; lng: number } | null>(null);
  const [riskData, setRiskData] = useState<Record<string, number> | null>(null);
  const [disasterType, setDisasterType] = useState<"flood" | "hurricane" | "wildfire">("flood");

  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnalyze = () => {
    if (!startDate || !endDate || !coordinates) return;
    setIsAnalyzing(true);
    console.log("Analyzing risk for:", { startDate, endDate, location, coordinates });
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <DashboardHeader />
      
      <main className="container mx-auto px-4 py-8 space-y-8">
        <div className="flex items-center gap-3 mb-6">
          <MapPin className="h-8 w-8 text-primary" />
          <div>
            <h2 className="text-3xl font-bold">Multi-Hazard Risk Assessment</h2>
            <p className="text-muted-foreground">Analyze and visualize risk across multiple hazards for any location</p>
          </div>
        </div>

        <InputForm
          startDate={startDate}
          endDate={endDate}
          location={location}
          coordinates={coordinates}
          disasterType={disasterType}
          onStartDateChange={setStartDate}
          onEndDateChange={setEndDate}
          onLocationChange={setLocation}
          onCoordinatesChange={setCoordinates}
          onDisasterTypeChange={setDisasterType}
          onAnalyze={handleAnalyze}
        />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <RiskMap 
              coordinates={coordinates} 
              isAnalyzing={isAnalyzing}
              disasterType={disasterType}
              startDate={startDate}
              endDate={endDate}
              onRiskDataUpdate={setRiskData}
            />
          </div>
          
          <div className="space-y-4">
            <RiskLegend />
            {isAnalyzing && riskData && (
              <div className="flex justify-end">
                <DownloadCsvButton riskData={riskData} disasterType={disasterType} />
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;

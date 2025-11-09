import { CalendarIcon, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Calendar } from "@/components/ui/calendar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { format } from "date-fns";
import { Card } from "@/components/ui/card";
import { toast } from "sonner";

interface InputFormProps {
  startDate?: Date;
  endDate?: Date;
  location: string;
  coordinates: { lat: number; lng: number } | null;
  disasterType: "flood" | "hurricane" | "wildfire";
  onStartDateChange: (date?: Date) => void;
  onEndDateChange: (date?: Date) => void;
  onLocationChange: (location: string) => void;
  onCoordinatesChange: (coords: { lat: number; lng: number } | null) => void;
  onDisasterTypeChange: (type: "flood" | "hurricane" | "wildfire") => void;
  onAnalyze: () => void;
}

const InputForm = ({
  startDate,
  endDate,
  location,
  coordinates,
  disasterType,
  onStartDateChange,
  onEndDateChange,
  onLocationChange,
  onCoordinatesChange,
  onDisasterTypeChange,
  onAnalyze,
}: InputFormProps) => {
  const handleLocationSearch = async () => {
    if (!location) return;
    
    const geocodePromise = async () => {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(location)}&limit=1`
      );
      const data = await response.json();
      
      if (data && data.length > 0) {
        const { lat, lon } = data[0];
        onCoordinatesChange({ lat: parseFloat(lat), lng: parseFloat(lon) });
        return data[0].display_name;
      } else {
        // Fallback to a default location if geocoding fails
        onCoordinatesChange({ lat: 40.7128, lng: -74.0060 });
        throw new Error("Location not found");
      }
    };
    
    toast.promise(geocodePromise(), {
      loading: "Finding location...",
      success: (displayName) => `Location found: ${displayName}`,
      error: "Location not found, using default (New York)",
    });
  };

  return (
    <Card className="p-6 bg-card border-border">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="space-y-2">
          <Label htmlFor="start-date">Start Date</Label>
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                className={cn(
                  "w-full justify-start text-left font-normal",
                  !startDate && "text-muted-foreground"
                )}
              >
                <CalendarIcon className="mr-2 h-4 w-4" />
                {startDate ? format(startDate, "PPP") : "Pick a date"}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar
                mode="single"
                selected={startDate}
                onSelect={onStartDateChange}
                initialFocus
                className="pointer-events-auto"
              />
            </PopoverContent>
          </Popover>
        </div>

        <div className="space-y-2">
          <Label htmlFor="end-date">End Date</Label>
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                className={cn(
                  "w-full justify-start text-left font-normal",
                  !endDate && "text-muted-foreground"
                )}
              >
                <CalendarIcon className="mr-2 h-4 w-4" />
                {endDate ? format(endDate, "PPP") : "Pick a date"}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar
                mode="single"
                selected={endDate}
                onSelect={onEndDateChange}
                initialFocus
                className="pointer-events-auto"
              />
            </PopoverContent>
          </Popover>
        </div>

        <div className="space-y-2">
          <Label htmlFor="location">Location Address</Label>
          <div className="flex gap-2">
            <Input
              id="location"
              placeholder="Enter address or pincode..."
              value={location}
              onChange={(e) => onLocationChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleLocationSearch();
                }
              }}
              className="flex-1"
            />
            <Button 
              size="icon" 
              variant="secondary"
              onClick={handleLocationSearch}
            >
              <Search className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      <div className="mt-6 flex flex-col md:flex-row justify-end items-start md:items-center gap-4">

        <div className="flex flex-col items-end gap-3">
          {location && !coordinates && (
            <p className="text-sm text-muted-foreground flex items-center gap-2">
              <Search className="h-4 w-4" />
              Click search to find location first
            </p>
          )}
          <Button 
            onClick={onAnalyze}
            className="bg-primary hover:bg-primary/90 text-primary-foreground"
            disabled={!startDate || !endDate || !coordinates}
          >
            Analyze Risk
          </Button>
        </div>
      </div>
    </Card>
  );
};

export default InputForm;

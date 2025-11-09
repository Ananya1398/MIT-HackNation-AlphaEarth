import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Download, FileImage, FileCode } from "lucide-react";
import { toast } from "sonner";

const DownloadControls = () => {
  const handleDownloadMap = (format: 'png' | 'html') => {
    toast.success(`Downloading map as ${format.toUpperCase()}...`);
    // Implementation would go here
  };

  const handleDownloadChart = (format: 'png' | 'html') => {
    toast.success(`Downloading chart as ${format.toUpperCase()}...`);
    // Implementation would go here
  };

  return (
    <Card className="p-6 bg-card border-border">
      <div className="flex items-center gap-2 mb-4">
        <Download className="h-5 w-5 text-primary" />
        <h3 className="text-lg font-semibold">Export Options</h3>
      </div>

      <div className="space-y-4">
        <div>
          <h4 className="text-sm font-medium mb-2">Risk Map</h4>
          <div className="flex gap-2">
            <Button 
              variant="outline" 
              size="sm" 
              className="flex-1"
              onClick={() => handleDownloadMap('png')}
            >
              <FileImage className="h-4 w-4 mr-2" />
              PNG
            </Button>
            <Button 
              variant="outline" 
              size="sm" 
              className="flex-1"
              onClick={() => handleDownloadMap('html')}
            >
              <FileCode className="h-4 w-4 mr-2" />
              HTML
            </Button>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-medium mb-2">Risk Chart</h4>
          <div className="flex gap-2">
            <Button 
              variant="outline" 
              size="sm" 
              className="flex-1"
              onClick={() => handleDownloadChart('png')}
            >
              <FileImage className="h-4 w-4 mr-2" />
              PNG
            </Button>
            <Button 
              variant="outline" 
              size="sm" 
              className="flex-1"
              onClick={() => handleDownloadChart('html')}
            >
              <FileCode className="h-4 w-4 mr-2" />
              HTML
            </Button>
          </div>
        </div>

        <div className="pt-4 border-t border-border">
          <Button 
            className="w-full bg-primary hover:bg-primary/90 text-primary-foreground"
            onClick={() => toast.success("Downloading complete dataset...")}
          >
            <Download className="h-4 w-4 mr-2" />
            Download All Data
          </Button>
        </div>
      </div>
    </Card>
  );
};

export default DownloadControls;

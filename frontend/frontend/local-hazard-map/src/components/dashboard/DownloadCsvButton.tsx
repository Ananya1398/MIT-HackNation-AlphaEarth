import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";
import { toast } from "sonner";

interface DownloadCsvButtonProps {
  riskData: Record<string, number> | null;
  disasterType: "flood" | "hurricane" | "wildfire";
}

const DownloadCsvButton = ({ riskData, disasterType }: DownloadCsvButtonProps) => {
  const handleDownload = () => {
    if (!riskData || Object.keys(riskData).length === 0) {
      toast.error("No risk data available to download");
      return;
    }

    // Convert risk data to CSV format
    const csvHeaders = "County,Risk Score,Disaster Type\n";
    const csvRows = Object.entries(riskData)
      .map(([county, score]) => `"${county}",${score.toFixed(2)},"${disasterType}"`)
      .join("\n");
    
    const csvContent = csvHeaders + csvRows;

    // Create blob and download
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    
    const capitalizedDisasterType = disasterType.charAt(0).toUpperCase() + disasterType.slice(1);
    link.setAttribute("href", url);
    link.setAttribute("download", `${capitalizedDisasterType}-risk-scores-${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = "hidden";
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    toast.success("CSV downloaded successfully");
  };

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={handleDownload}
      disabled={!riskData || Object.keys(riskData).length === 0}
      className="gap-2"
    >
      <Download className="h-4 w-4" />
      Download CSV
    </Button>
  );
};

export default DownloadCsvButton;

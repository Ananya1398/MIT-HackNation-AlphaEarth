import { Activity } from "lucide-react";

const DashboardHeader = () => {
  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Activity className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h1 className="text-xl font-bold">Risk Dashboard</h1>
            <p className="text-sm text-muted-foreground">Real-time hazard analysis</p>
          </div>
        </div>
      </div>
    </header>
  );
};

export default DashboardHeader;

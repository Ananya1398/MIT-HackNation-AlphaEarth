import { Card } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from "recharts";
import { TrendingUp } from "lucide-react";

interface RiskChartProps {
  startDate?: Date;
  endDate?: Date;
}

const RiskChart = ({ startDate, endDate }: RiskChartProps) => {
  // Mock data - in production, this would come from the backend
  const data = [
    { date: "Jan 1", riskScore: 4.2 },
    { date: "Jan 5", riskScore: 5.1 },
    { date: "Jan 10", riskScore: 6.8 },
    { date: "Jan 15", riskScore: 7.5 },
    { date: "Jan 20", riskScore: 6.2 },
    { date: "Jan 25", riskScore: 8.1 },
    { date: "Jan 30", riskScore: 7.3 },
    { date: "Feb 4", riskScore: 5.9 },
    { date: "Feb 9", riskScore: 4.5 },
    { date: "Feb 14", riskScore: 6.7 },
  ];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm font-medium mb-1">{payload[0].payload.date}</p>
          <p className="text-sm text-muted-foreground">
            Risk Score: <span className="font-semibold text-primary">{payload[0].value.toFixed(2)}</span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="p-6 bg-card border-border">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold">Daily Risk Score Progression</h3>
          <p className="text-sm text-muted-foreground mt-1">
            {startDate && endDate ? 
              `${startDate.toLocaleDateString()} - ${endDate.toLocaleDateString()}` : 
              "Select dates to view progression"
            }
          </p>
        </div>
        <TrendingUp className="h-5 w-5 text-primary" />
      </div>

      <div className="h-[350px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis 
              dataKey="date" 
              stroke="hsl(var(--muted-foreground))"
              style={{ fontSize: '12px' }}
            />
            <YAxis 
              stroke="hsl(var(--muted-foreground))"
              style={{ fontSize: '12px' }}
              domain={[0, 10]}
              ticks={[0, 2, 4, 6, 8, 10]}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area 
              type="monotone" 
              dataKey="riskScore" 
              stroke="hsl(var(--primary))" 
              strokeWidth={3}
              fill="url(#riskGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
};

export default RiskChart;

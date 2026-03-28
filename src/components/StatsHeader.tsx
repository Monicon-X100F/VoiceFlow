import { useEffect, useState } from "react";
import { Flame, FileText, Type, Settings2, Languages, Cpu, Mic, Zap } from "lucide-react";
import { api } from "@/lib/api";
import type { Stats } from "@/lib/types";

type ConfigData = {
  model: string;
  language: string;
  micName: string;
  computeDevice: string;
  isUsingGpu: boolean;
};

export function StatsHeader() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [config, setConfig] = useState<ConfigData | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        // Load stats
        const statsData = await api.getStats();
        setStats(statsData);

        // Load config (settings + options + GPU info)
        const [settings, options, gpuInfo] = await Promise.all([
          api.getSettings(),
          api.getOptions(),
          api.getGpuInfo(),
        ]);

        // Find mic name
        const currentMic = options.microphones.find(
          (m) => m.id === settings.microphone
        );
        const micName =
          settings.microphone === -1
            ? "System Default"
            : currentMic?.name || "Unknown Mic";

        // Determine compute device
        const isUsingGpu = settings.device === "cuda" || (settings.device === "auto" && gpuInfo.cudaAvailable);
        const computeDevice = isUsingGpu
          ? gpuInfo.gpuName?.replace("NVIDIA ", "").replace(" Laptop GPU", "") || "CUDA GPU"
          : "CPU";

        setConfig({
          model: settings.model,
          language: settings.language,
          micName: micName,
          computeDevice,
          isUsingGpu,
        });
      } catch (error) {
        console.error("Failed to load data:", error);
        setStats({
          totalTranscriptions: 0,
          totalWords: 0,
          totalCharacters: 0,
          streakDays: 0,
        });
      }
    };
    load();
  }, []);

  if (!stats) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="md:col-span-2 h-56 rounded-3xl bg-muted/20 animate-pulse" />
        <div className="h-56 rounded-3xl bg-muted/20 animate-pulse" />
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      {/* Hero Card - Words */}
      <div className="md:col-span-2 glass-card overflow-hidden relative group">

        <div className="p-8 relative z-10 flex flex-col h-full justify-between min-h-[220px]">
          <div className="space-y-3">
            <div className="badge-glow w-fit">
              <Type className="w-3.5 h-3.5" />
              <span className="text-xs font-medium uppercase tracking-wider">
                Total Words Dictated
              </span>
            </div>
            <h2 className="text-5xl md:text-7xl font-bold text-foreground tracking-tighter">
              {stats.totalWords.toLocaleString()}
              <span className="headline-serif text-primary/60 text-4xl md:text-5xl ml-2">
                words
              </span>
            </h2>
          </div>

          <div className="mt-8 flex gap-4 flex-wrap">
            {/* Streak */}
            <div className="glass-strong flex items-center gap-3 rounded-2xl px-5 py-3">
              <div className="p-2 bg-primary/15 rounded-xl text-primary">
                <Flame className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground font-semibold uppercase tracking-widest mb-0.5">
                  Current Streak
                </p>
                <p className="text-lg text-foreground font-bold leading-none">
                  {stats.streakDays}{" "}
                  <span className="text-sm font-normal text-muted-foreground">
                    days
                  </span>
                </p>
              </div>
            </div>

            {/* Notes */}
            <div className="glass-strong flex items-center gap-3 rounded-2xl px-5 py-3">
              <div className="p-2 bg-primary/10 rounded-xl text-primary/70">
                <FileText className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground font-semibold uppercase tracking-widest mb-0.5">
                  Total Notes
                </p>
                <p className="text-lg text-foreground font-bold leading-none">
                  {stats.totalTranscriptions}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Active Configuration Card */}
      <div className="glass-card flex flex-col overflow-hidden relative">

        <div className="p-6 h-full flex flex-col relative z-10">
          <div className="flex items-center justify-between mb-6">
            <div className="p-2.5 rounded-xl bg-primary/10 border border-primary/20 text-primary">
              <Settings2 className="w-5 h-5" />
            </div>
            <span className="text-xs font-mono text-muted-foreground/60 bg-secondary/50 px-3 py-1.5 rounded-full">
              Active Config
            </span>
          </div>

          <div className="space-y-5 mt-auto">
            {/* Model */}
            <ConfigItem
              icon={Cpu}
              label="Model"
              value={
                config?.model
                  ? config.model.charAt(0).toUpperCase() + config.model.slice(1)
                  : "Loading..."
              }
            />

            {/* Language */}
            <ConfigItem
              icon={Languages}
              label="Language"
              value={
                config?.language
                  ? config.language === "auto"
                    ? "Auto-Detect"
                    : config.language.toUpperCase()
                  : "Loading..."
              }
            />

            {/* Microphone */}
            <ConfigItem
              icon={Mic}
              label="Input"
              value={config?.micName || "Loading..."}
              truncate
            />

            {/* Compute Device */}
            <ConfigItem
              icon={config?.isUsingGpu ? Zap : Cpu}
              label="Compute"
              value={config?.computeDevice || "Loading..."}
              highlight={config?.isUsingGpu}
              truncate
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function ConfigItem({
  icon: Icon,
  label,
  value,
  truncate = false,
  highlight = false,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  truncate?: boolean;
  highlight?: boolean;
}) {
  return (
    <div className="group/item">
      <div className="flex items-center gap-1.5 mb-1.5">
        <Icon className={`w-3 h-3 ${highlight ? "text-primary" : "text-muted-foreground/60"}`} />
        <span className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
          {label}
        </span>
      </div>
      <div
        className={`text-sm font-semibold tracking-tight pl-4 border-l-2 transition-colors ${truncate ? "truncate" : ""} ${
          highlight
            ? "text-primary border-primary/40 group-hover/item:border-primary"
            : "text-foreground border-primary/20 group-hover/item:border-primary"
        }`}
        title={truncate ? value : undefined}
      >
        {value}
      </div>
    </div>
  );
}

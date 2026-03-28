import { useEffect, useState } from "react";
import { Copy, Trash2, Clock, CalendarDays, Search, Mic, FileAudio } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { StatsHeader } from "@/components/StatsHeader";
import { api } from "@/lib/api";
import type { HistoryEntry } from "@/lib/types";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { base64ToBlobUrl, revokeUrl, isInvalidAudioPayload } from "@/lib/audio";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

export function HomePage() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [showPlayer, setShowPlayer] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioMeta, setAudioMeta] = useState<{ fileName?: string; mime?: string; durationMs?: number } | null>(null);
  const [loadingAudioFor, setLoadingAudioFor] = useState<number | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        setError(null);
        const data = await api.getHistory(50, 0, undefined, false);
        setHistory(data);
      } catch (error) {
        console.error("Failed to load history:", error);
        setError("Failed to load history. Please try again.");
        toast.error("Failed to load history");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  useEffect(() => {
    return () => revokeUrl(audioUrl);
  }, [audioUrl]);

  const handleCopy = async (text: string) => {
    try {
      await api.copyToClipboard(text);
      toast.success("Copied to clipboard");
    } catch {
      try {
        await navigator.clipboard.writeText(text);
        toast.success("Copied to clipboard");
      } catch {
        toast.error("Failed to copy to clipboard");
      }
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await api.deleteHistory(id);
      setHistory((prev) => prev.filter((h) => h.id !== id));
      toast.success("Transcription deleted");
    } catch (error) {
      console.error("Failed to delete:", error);
      toast.error("Failed to delete transcription");
    }
  };

  const handlePlayAudio = async (historyId: number) => {
    setLoadingAudioFor(historyId);
    try {
      const response = await api.getHistoryAudio(historyId);
      revokeUrl(audioUrl);
      const url = base64ToBlobUrl(response.base64, response.mime);
      setAudioUrl(url);
      setAudioMeta({
        fileName: response.fileName,
        mime: response.mime,
        durationMs: response.durationMs,
      });
      setShowPlayer(true);
    } catch (error) {
      console.error("Failed to load audio recording:", error);
      toast.error(isInvalidAudioPayload(error) ? "Audio file is corrupted" : "Audio file not found");
      revokeUrl(audioUrl);
      setAudioUrl(null);
      setShowPlayer(false);
      setAudioMeta(null);
    } finally {
      setLoadingAudioFor(null);
    }
  };

  const filteredHistory = history.filter((entry) =>
    entry.text.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const groupedHistory = groupByDate(filteredHistory);
  const durationMs = audioMeta?.durationMs;

  return (
    <>
    <div className="min-h-screen w-full bg-background/50 relative overflow-x-hidden">
      {/* Background effects */}
      <div className="fixed inset-0 bg-grid opacity-20 pointer-events-none overflow-hidden" />
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="orb orb-primary w-[500px] h-[500px] absolute -top-60 -right-60 opacity-20" />
        <div className="orb orb-secondary w-[400px] h-[400px] absolute bottom-0 -left-40 opacity-15" />
      </div>

      <div className="w-full max-w-[1600px] mx-auto p-6 md:p-10 space-y-10 relative z-10">
        {/* Header Section */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tighter text-foreground mb-2">
              Dashboard
            </h1>
            <p className="text-lg text-muted-foreground/80 font-light max-w-2xl">
              Your voice, organized. Manage your recent transcriptions and
              insights.
            </p>
          </div>
          <div className="w-full md:w-auto flex items-center gap-3">
            <div className="relative group">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
              <Input
                placeholder="Search transcriptions..."
                className="pl-11 w-full md:w-[320px] h-12 bg-background/50 border-border/50 rounded-xl focus:bg-background focus:border-primary/30 transition-all"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </div>
        </div>

        {/* Stats / Hero Section */}
        <section>
          <StatsHeader />
        </section>

        {/* Divider */}
        <div className="divider-gradient" />

        {/* Recent History Grid */}
        <section className="space-y-8">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-semibold tracking-tight flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10 text-primary">
                <Clock className="w-5 h-5" />
              </div>
              Recent History
            </h2>
            <span className="badge-glow">
              {filteredHistory.length} entries
            </span>
          </div>

          {loading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[...Array(6)].map((_, i) => (
                <div
                  key={i}
                  className="h-48 rounded-2xl bg-muted/20 animate-pulse"
                />
              ))}
            </div>
          ) : error ? (
            <div className="text-center py-20 glass-card">
              <p className="text-destructive font-medium mb-4">{error}</p>
              <Button
                variant="outline"
                onClick={() => window.location.reload()}
                className="rounded-xl"
              >
                Try again
              </Button>
            </div>
          ) : Object.keys(groupedHistory).length === 0 ? (
            <div className="flex flex-col items-center justify-center py-24 text-center space-y-6 glass-card">
              <div className="relative">
                <div className="w-24 h-24 rounded-3xl bg-primary/10 flex items-center justify-center">
                  <Mic className="w-12 h-12 text-primary/40" />
                </div>
                <div className="absolute -top-2 -right-2 w-8 h-8 rounded-full bg-muted/50 flex items-center justify-center">
                  <Search className="w-4 h-4 text-muted-foreground/50" />
                </div>
              </div>
              <div className="space-y-2">
                <p className="text-xl font-medium text-foreground">
                  No transcriptions{" "}
                  <span className="headline-serif text-muted-foreground">
                    yet
                  </span>
                </p>
                <p className="text-muted-foreground">
                  {searchQuery
                    ? "Try adjusting your search terms."
                    : "Press Ctrl+Win to start your first dictation."}
                </p>
              </div>
            </div>
          ) : (
            <div className="space-y-10">
              {Object.entries(groupedHistory).map(([dateLabel, entries]) => (
                <div key={dateLabel} className="space-y-4">
                  <div className="flex items-center gap-3 sticky top-0 z-10 py-3 bg-background/95 backdrop-blur-sm">
                    <CalendarDays className="w-4 h-4 text-primary/60" />
                    <h3 className="text-sm font-bold uppercase tracking-widest text-muted-foreground">
                      {dateLabel}
                    </h3>
                    <div className="h-px flex-1 bg-gradient-to-r from-border/60 to-transparent" />
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
                    {entries.map((entry) => (
                      <HistoryCard
                        key={entry.id}
                        entry={entry}
                        onCopy={handleCopy}
                        onDelete={handleDelete}
                        onPlayAudio={handlePlayAudio}
                        isLoadingAudio={loadingAudioFor === entry.id}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>

      <Dialog
        open={showPlayer}
        onOpenChange={(open) => {
          setShowPlayer(open);
          if (!open) {
            revokeUrl(audioUrl);
            setAudioUrl(null);
            setAudioMeta(null);
          }
        }}
      >
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>Audio Recording</DialogTitle>
          <DialogDescription>Playback of the recorded audio for this transcription</DialogDescription>
        </DialogHeader>
        {audioUrl ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <FileAudio className="w-4 h-4 text-primary" />
                <span>{audioMeta?.fileName || "history_audio.wav"}</span>
              </div>
              {durationMs ? <span>{Math.round(durationMs / 1000)}s</span> : null}
            </div>
            {/* biome-ignore lint/a11y/useMediaCaption: transcript text is already displayed in the history card */}
            <audio controls autoPlay className="w-full">
              <source src={audioUrl} type={audioMeta?.mime || "audio/wav"} />
              Your browser does not support audio playback.
            </audio>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">No audio loaded.</p>
        )}
      </DialogContent>
    </Dialog>
    </>
  );
}

function HistoryCard({
  entry,
  onCopy,
  onDelete,
  onPlayAudio,
  isLoadingAudio,
}: {
  entry: HistoryEntry;
  onCopy: (text: string) => void;
  onDelete: (id: number) => void;
  onPlayAudio: (id: number) => void;
  isLoadingAudio: boolean;
}) {
  const hasAudio = !!entry.has_audio;
  return (
    <div className="group glass-card flex flex-col justify-between h-full p-5 transition-colors duration-150">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-muted-foreground bg-secondary/50 px-2.5 py-1 rounded-lg">
            {formatTime(entry.created_at)}
          </span>
          {hasAudio && (
            <Badge variant="secondary" className="text-[11px] flex items-center gap-1">
              <FileAudio className="w-3 h-3" />
              Audio
            </Badge>
          )}
        </div>
        <div className="opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-muted-foreground hover:text-primary hover:bg-primary/10 rounded-lg"
            onClick={() => onCopy(entry.text)}
            aria-label="Copy transcription"
          >
            <Copy className="h-3.5 w-3.5" />
          </Button>
          {hasAudio && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-muted-foreground hover:text-primary hover:bg-primary/10 rounded-lg"
              onClick={() => onPlayAudio(entry.id)}
              disabled={isLoadingAudio}
              aria-label="Play audio recording"
            >
              <FileAudio className="h-3.5 w-3.5" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded-lg"
            onClick={() => onDelete(entry.id)}
            aria-label="Delete transcription"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* Content */}
      <p className="text-base leading-relaxed line-clamp-4 font-medium text-foreground/90 group-hover:text-foreground transition-colors flex-grow">
        {entry.text}
      </p>

      {/* Footer */}
      <div className="mt-4 pt-3 border-t border-border/30">
        <div className="flex items-center justify-between">
          <div className="text-[10px] uppercase tracking-wider font-semibold text-primary/50 group-hover:text-primary/80 transition-colors">
            {entry.word_count} words
          </div>
          {hasAudio && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-xs text-primary hover:text-primary hover:bg-primary/10 transition-colors"
              onClick={() => onPlayAudio(entry.id)}
              disabled={isLoadingAudio}
            >
              <FileAudio className="w-3 h-3 mr-1" />
              {isLoadingAudio ? "Loading..." : "Play"}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

function formatTime(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function groupByDate(entries: HistoryEntry[]): Record<string, HistoryEntry[]> {
  const groups: Record<string, HistoryEntry[]> = {};
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  for (const entry of entries) {
    const entryDate = new Date(entry.created_at);
    let label: string;

    if (isSameDay(entryDate, today)) {
      label = "Today";
    } else if (isSameDay(entryDate, yesterday)) {
      label = "Yesterday";
    } else {
      label = entryDate.toLocaleDateString([], {
        weekday: "long",
        month: "long",
        day: "numeric",
      });
    }

    if (!groups[label]) {
      groups[label] = [];
    }
    groups[label].push(entry);
  }

  return groups;
}

function isSameDay(d1: Date, d2: Date): boolean {
  return (
    d1.getFullYear() === d2.getFullYear() &&
    d1.getMonth() === d2.getMonth() &&
    d1.getDate() === d2.getDate()
  );
}

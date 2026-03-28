import { useEffect, useState } from "react";
import { Search, Copy, Trash2, CalendarDays, Clock, Mic, FileAudio } from "lucide-react";
import { toast } from "sonner";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { base64ToBlobUrl, revokeUrl, isInvalidAudioPayload } from "@/lib/audio";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { api } from "@/lib/api";
import type { HistoryEntry } from "@/lib/types";

export function HistoryPage() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [showPlayer, setShowPlayer] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioMeta, setAudioMeta] = useState<{ fileName?: string; mime?: string; durationMs?: number } | null>(null);
  const [loadingAudioFor, setLoadingAudioFor] = useState<number | null>(null);

  // Reusing the same load logic as HomePage for consistency
  const loadHistory = async (searchQuery?: string) => {
    setLoading(true);
    try {
      // Fetch 100 items by default for the full page view
      const data = await api.getHistory(100, 0, searchQuery || undefined, false);
      setHistory(data);
    } catch (error) {
      console.error("Failed to load history:", error);
      toast.error("Failed to load history");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  useEffect(() => {
    const debounce = setTimeout(() => {
      loadHistory(search);
    }, 500);
    return () => clearTimeout(debounce);
  }, [search]);

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

  const groupedHistory = groupByDate(history);
  const durationMs = audioMeta?.durationMs;

  return (
    <>
    <div className="min-h-screen w-full bg-background/50 relative overflow-x-hidden">
      {/* Background effects */}
      <div className="fixed inset-0 bg-grid opacity-20 pointer-events-none overflow-hidden" />
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="orb orb-secondary w-[450px] h-[450px] absolute -top-40 -left-40 opacity-15" />
        <div className="orb orb-primary w-[350px] h-[350px] absolute bottom-20 -right-40 opacity-20" />
      </div>

      <div className="w-full max-w-[1600px] mx-auto p-6 md:p-10 space-y-10 relative z-10">

        {/* Header Section */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tighter text-foreground mb-2">
              Full History
            </h1>
            <p className="text-lg text-muted-foreground/80 font-light max-w-2xl">
              A complete archive of your voice notes and dictations.
            </p>
          </div>
          
          <div className="w-full md:w-[400px] relative group">
             <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors" />
             <Input 
               placeholder="Search archive..." 
               className="pl-10 h-11 w-full bg-background/50 border-border/50 focus:bg-background transition-all shadow-sm"
               value={search}
               onChange={(e) => setSearch(e.target.value)}
             />
          </div>
        </div>

        {/* Content */}
        <section className="min-h-[500px]">
          {loading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
               {[...Array(12)].map((_, i) => (
                 <div key={i} className="h-48 rounded-xl bg-secondary/20 animate-pulse" />
               ))}
            </div>
          ) : Object.keys(groupedHistory).length === 0 ? (
            <div className="flex flex-col items-center justify-center py-32 text-center space-y-6 border border-dashed border-border/50 rounded-3xl bg-secondary/5">
               <div className="relative">
                 <div className="w-24 h-24 rounded-3xl bg-primary/10 flex items-center justify-center">
                   <Mic className="w-12 h-12 text-primary/40" />
                 </div>
                 <div className="absolute -top-2 -right-2 w-8 h-8 rounded-full bg-muted/50 flex items-center justify-center">
                   <Search className="w-4 h-4 text-muted-foreground/50" />
                 </div>
               </div>
               <div className="space-y-1">
                 <p className="text-xl font-medium text-foreground">
                    {search ? "No matching results" : "Archive is empty"}
                 </p>
                 <p className="text-muted-foreground">
                   {search ? "Try searching for simpler keywords." : "Everything you transcribe will be saved here."}
                 </p>
               </div>
            </div>
          ) : (
            <div className="space-y-12">
              {Object.entries(groupedHistory).map(([dateLabel, entries]) => (
                <div key={dateLabel} className="space-y-4">
                  <div className="flex items-center gap-3 sticky top-0 z-10 bg-background/95 backdrop-blur py-3 text-sm font-bold uppercase tracking-widest text-muted-foreground">
                    <CalendarDays className="w-4 h-4 text-primary" />
                    <h3>{dateLabel}</h3>
                    <div className="h-px flex-1 bg-border/40" />
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                    {entries.map((entry) => {
                      const hasAudio = !!entry.has_audio;
                      return (
                        <Card
                          key={entry.id}
                          className="group flex flex-col justify-between h-full bg-card/60 backdrop-blur-sm border-border/50 hover:bg-card hover:border-primary/20 transition-colors duration-150"
                        >
                          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-mono text-muted-foreground bg-secondary/50 px-2 py-1 rounded flex items-center gap-1.5">
                                <Clock className="w-3 h-3" />
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
                                size="icon-sm"
                                className="h-7 w-7 text-muted-foreground hover:text-primary hover:bg-primary/10"
                                onClick={() => handleCopy(entry.text)}
                                aria-label="Copy transcription"
                              >
                                <Copy className="h-3.5 w-3.5" />
                              </Button>
                              {hasAudio && (
                                <Button
                                  variant="ghost"
                                  size="icon-sm"
                                  className="h-7 w-7 text-muted-foreground hover:text-primary hover:bg-primary/10"
                                  onClick={() => handlePlayAudio(entry.id)}
                                  disabled={loadingAudioFor === entry.id}
                                  aria-label="Play audio recording"
                                >
                                  <FileAudio className="h-3.5 w-3.5" />
                                </Button>
                              )}
                              <Button
                                variant="ghost"
                                size="icon-sm"
                                className="h-7 w-7 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                                onClick={() => handleDelete(entry.id)}
                                aria-label="Delete transcription"
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                              </Button>
                            </div>
                          </CardHeader>
                          <CardContent className="pt-2 flex-grow">
                            <p className="text-base leading-relaxed line-clamp-6 font-medium text-foreground/90 group-hover:text-foreground transition-colors">
                              {entry.text}
                            </p>
                          </CardContent>
                          <div className="px-6 pb-4 pt-0 mt-auto flex items-center justify-between">
                             <div className="text-[10px] uppercase tracking-wider font-semibold text-primary/40 group-hover:text-primary/80 transition-colors">
                                {entry.word_count} words
                             </div>
                             {hasAudio && (
                               <Button
                                 variant="ghost"
                                 size="sm"
                                 className="h-6 px-2 text-xs text-primary hover:text-primary hover:bg-primary/10 transition-colors"
                                 onClick={() => handlePlayAudio(entry.id)}
                                 disabled={loadingAudioFor === entry.id}
                               >
                                 <FileAudio className="w-3 h-3 mr-1" />
                                 {loadingAudioFor === entry.id ? "Loading..." : "Play"}
                               </Button>
                             )}
                          </div>
                        </Card>
                      );
                    })}
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
      label = entryDate.toLocaleDateString([], { weekday: 'long', month: "long", day: "numeric" });
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

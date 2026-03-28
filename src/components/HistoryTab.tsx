import { useEffect, useState } from "react";
import { Search, Copy, Trash2, CalendarDays, Clock, Mic } from "lucide-react";
import { toast } from "sonner";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { api } from "@/lib/api";
import type { HistoryEntry } from "@/lib/types";

export function HistoryTab() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);

  const loadHistory = async (searchQuery?: string) => {
    setLoading(true);
    try {
      // Fetch more items for the history tab since it's the main view
      const data = await api.getHistory(100, 0, searchQuery || undefined);
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
    }, 500); // 500ms debounce
    return () => clearTimeout(debounce);
  }, [search]);

  const handleCopy = async (text: string) => {
    try {
      await api.copyToClipboard(text);
      toast.success("Copied to clipboard");
    } catch {
      // Fallback
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

  const groupedHistory = groupByDate(history);

  return (
    <div className="min-h-screen w-full bg-background/50">
      <div className="w-full max-w-[1600px] mx-auto p-6 md:p-10 space-y-8">
        
        {/* Header & Search */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
           <div>
              <h1 className="text-4xl font-bold tracking-tighter text-foreground mb-2">History</h1>
              <p className="text-muted-foreground font-light text-lg">
                Browse and manage your past transcriptions.
              </p>
           </div>
           
           <div className="w-full md:w-[400px] relative group">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors" />
              <Input
                type="search"
                placeholder="Search history..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-10 h-11 rounded-xl bg-background/50 border-border/50 shadow-sm focus-visible:ring-primary/20 hover:bg-background transition-all"
              />
           </div>
        </div>

        {/* Content */}
        <div className="min-h-[500px]">
          {loading ? (
             <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {[...Array(8)].map((_, i) => (
                  <div key={i} className="h-40 rounded-xl bg-secondary/20 animate-pulse" />
                ))}
             </div>
          ) : Object.keys(groupedHistory).length === 0 ? (
            <div className="flex flex-col items-center justify-center py-24 text-center space-y-6 border border-dashed border-border/50 rounded-3xl bg-secondary/5 mt-8">
               <div className="relative">
                 <div className="w-20 h-20 rounded-2xl bg-primary/10 flex items-center justify-center">
                   <Mic className="w-10 h-10 text-primary/40" />
                 </div>
                 <div className="absolute -top-1 -right-1 w-6 h-6 rounded-full bg-muted/50 flex items-center justify-center">
                   <Search className="w-3 h-3 text-muted-foreground/50" />
                 </div>
               </div>
               <div>
                  <p className="text-lg font-medium text-foreground">
                    {search ? "No matches found" : "No history yet"}
                  </p>
                  <p className="text-muted-foreground text-sm mt-1">
                    {search ? "Try a different search term" : "Your transcriptions will appear here"}
                  </p>
               </div>
            </div>
          ) : (
            <div className="space-y-12">
              {Object.entries(groupedHistory).map(([dateLabel, entries]) => (
                <div key={dateLabel} className="space-y-5">
                   {/* Date Header */}
                   <div className="flex items-center gap-3 sticky top-0 z-10 bg-background/95 backdrop-blur py-3">
                      <CalendarDays className="w-4 h-4 text-primary" />
                      <h3 className="text-sm font-bold uppercase tracking-widest text-muted-foreground">
                        {dateLabel}
                      </h3>
                      <div className="h-px flex-1 bg-border/40" />
                   </div>

                   {/* Grid */}
                   <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                      {entries.map((entry) => (
                        <Card
                          key={entry.id}
                          className="group flex flex-col h-full bg-card/60 backdrop-blur-sm border-border/50 shadow-sm hover:bg-card hover:border-primary/20 transition-colors duration-150"
                        >
                          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                             <span className="text-xs font-mono text-muted-foreground bg-secondary/50 px-2 py-1 rounded flex items-center gap-1.5">
                                <Clock className="w-3 h-3" />
                                {formatTime(entry.created_at)}
                             </span>
                             <div className="opacity-0 group-hover:opacity-100 transition-all flex gap-1">
                                <Button
                                  variant="ghost"
                                  size="icon-sm"
                                  className="h-7 w-7 text-muted-foreground hover:text-primary hover:bg-primary/10 transition-colors"
                                  onClick={() => handleCopy(entry.text)}
                                >
                                  <Copy className="h-3.5 w-3.5" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="icon-sm"
                                  className="h-7 w-7 text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                                  onClick={() => handleDelete(entry.id)}
                                >
                                  <Trash2 className="h-3.5 w-3.5" />
                                </Button>
                             </div>
                          </CardHeader>
                          <CardContent className="pt-2 flex-grow flex flex-col gap-4">
                             <p className="text-sm md:text-base font-medium leading-relaxed text-foreground/90 group-hover:text-foreground line-clamp-5 transition-colors">
                               {entry.text}
                             </p>
                             <div className="mt-auto">
                                <span className="text-[10px] uppercase tracking-wider font-semibold text-primary/40 group-hover:text-primary/70 transition-colors">
                                   {entry.word_count} words
                                </span>
                             </div>
                          </CardContent>
                        </Card>
                      ))}
                   </div>
                </div>
              ))}
            </div>
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

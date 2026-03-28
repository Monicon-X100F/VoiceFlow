import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ModelDownloadProgress } from "./ModelDownloadProgress";

interface ModelDownloadModalProps {
  open: boolean;
  modelName: string;
  onComplete: (success: boolean) => void;
  onCancel: () => void;
}

export function ModelDownloadModal({
  open,
  modelName,
  onComplete,
  onCancel,
}: ModelDownloadModalProps) {
  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onCancel()}>
      <DialogContent
        className="sm:max-w-md glass-strong rounded-2xl border-border/50"
        onPointerDownOutside={(e) => e.preventDefault()}
        onEscapeKeyDown={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle>Downloading Model</DialogTitle>
          <DialogDescription>
            Downloading the {modelName} model for transcription
          </DialogDescription>
        </DialogHeader>
        <div className="flex items-center justify-center py-4">
          <ModelDownloadProgress
            modelName={modelName}
            onComplete={onComplete}
            onCancel={onCancel}
            autoStart={true}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}

# Task 3: Update Frontend Types and API

## Objective

Add the new GPU vendor fields to the TypeScript `GpuInfo` interface so the frontend can consume AMD GPU status information from the backend.

## Context

The backend `get_gpu_info()` RPC now returns three new fields: `gpuVendor`, `rocmAvailable`, and `rocmMessage` (added in Task 2). The frontend TypeScript types need to match.

The frontend calls the backend via `pyloid-js` RPC:
```typescript
const gpuInfo = await api.getGpuInfo(); // calls rpc.call("get_gpu_info")
```

The `api.ts` file doesn't need changes — it just passes through whatever the backend returns. Only the type definition needs updating.

## File to Modify

- `src/lib/types.ts` — update `GpuInfo` interface (currently at line 87)

## Requirements

Update the `GpuInfo` interface from:

```typescript
export interface GpuInfo {
  cudaAvailable: boolean;
  deviceCount: number;
  gpuName: string | null;
  supportedComputeTypes: string[];
  currentDevice: string;
  currentComputeType: string;
  cudnnAvailable: boolean;
  cudnnMessage: string | null;
}
```

To:

```typescript
export interface GpuInfo {
  cudaAvailable: boolean;
  deviceCount: number;
  gpuName: string | null;
  supportedComputeTypes: string[];
  currentDevice: string;
  currentComputeType: string;
  cudnnAvailable: boolean;
  cudnnMessage: string | null;
  gpuVendor: "nvidia" | "amd" | null;
  rocmAvailable: boolean;
  rocmMessage: string | null;
}
```

## What NOT to Change

- Do NOT modify `api.ts` — it already passes through all backend fields
- Do NOT modify any component files (that's Tasks 4-5)
- Do NOT add any new interfaces or types beyond what's specified
- Do NOT change any other existing interfaces

## Verification Steps

Before considering this task complete, verify:

1. **TypeScript compiles**: Run `cd /home/user/VoiceFlow && pnpm run lint` — must pass with no new errors related to `GpuInfo`
2. **Type is correct**: Read `src/lib/types.ts` and confirm the `GpuInfo` interface has exactly 11 fields (8 original + 3 new)
3. **Union type is correct**: The `gpuVendor` field must use a string literal union `"nvidia" | "amd" | null`, not just `string | null`
4. **No other changes**: Verify no other interfaces or types were modified
